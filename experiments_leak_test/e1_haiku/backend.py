"""E1 (Claude Haiku 5.5 reader): configuration loading and a hardened Anthropic backend.

PREPARED 2026-10-19, NOT RUN. Nothing in this file makes an API call at import time.

What this adds on top of `run_reader.AnthropicBackend` (which already does the client setup, the
prompt rendering, the token guard, the max_tokens doubling retry and the append-only cache):

  1. per-item error isolation  - a permanent API error on one item records an invalid answer for
     that item instead of aborting the whole run (the cache keeps everything already answered);
  2. an explicit rate-limit layer - applied only after the SDK's own retries are exhausted,
     honouring the `retry-after` header, bounded by a total wait budget;
  3. a call-count ceiling      - a second guard beside the pre-registered token guard;
  4. a structured call log     - one JSON line per API call (request id, tokens, latency, outcome)
     so a finished run can be audited and the spend reconciled against the console.

None of this changes the prompts, the items, the parsing, the output schema or the decision rule.
`run_reader.py` is NOT modified: `run_e1.py` rebinds `run_reader.AnthropicBackend` to the class
below before calling `run_reader.main()`, the same convention the project already uses to keep the
upstream repo unmodified (experiment_plan_RQ0.md, decision F2).
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEAK = HERE.parent
if str(LEAK) not in sys.path:
    sys.path.insert(0, str(LEAK))

import run_reader  # noqa: E402  (prompts, rendering, parsing, cache: single source of truth)
from run_reader import Raw, Request  # noqa: E402

CONFIG_PATH = HERE / "e1_config.json"

# Exception names looked up on the SDK module at runtime, so a missing class in an older SDK (or in
# the stub used by the mock tests) degrades gracefully instead of raising AttributeError.
_PERMANENT = ("BadRequestError", "AuthenticationError", "PermissionDeniedError",
              "NotFoundError", "UnprocessableEntityError")
_NETWORK = ("APIConnectionError", "APITimeoutError")


class _Unreachable(Exception):
    """Placeholder for an SDK exception class this SDK version does not define."""


@dataclass
class E1Config:
    model: str
    effort: str
    max_tokens_total: int
    max_calls: int
    workers: int
    timeout_s: float
    sdk_max_retries: int
    rate_limit_extra_attempts: int
    rate_limit_max_wait_s: float
    network_extra_attempts: int
    items_path: Path
    items_sha256: str
    n_items: int
    answers_path: Path
    cache_path: Path
    call_log_path: Path
    price_in: float
    price_out: float
    stages: list
    raw: dict

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "E1Config":
        cfg = json.loads(path.read_text(encoding="utf-8"))
        out = cfg["outputs"]
        return cls(
            model=cfg["model"]["id"],
            effort=cfg["request"]["output_config"]["effort"],
            max_tokens_total=cfg["guards"]["max_tokens_total"],
            max_calls=cfg["guards"]["max_calls"],
            workers=cfg["run"]["workers"],
            timeout_s=cfg["client"]["timeout_s"],
            sdk_max_retries=cfg["client"]["sdk_max_retries"],
            rate_limit_extra_attempts=cfg["client"]["rate_limit_extra_attempts"],
            rate_limit_max_wait_s=cfg["client"]["rate_limit_max_wait_s"],
            network_extra_attempts=cfg["client"]["network_extra_attempts"],
            items_path=(path.parent / cfg["items"]["path"]).resolve(),
            items_sha256=cfg["items"]["sha256"],
            n_items=cfg["items"]["n_items"],
            answers_path=(path.parent / out["answers"]).resolve(),
            cache_path=(path.parent / out["cache"]).resolve(),
            call_log_path=(path.parent / out["call_log"]).resolve(),
            price_in=cfg["model"]["price_per_mtok_input_usd"],
            price_out=cfg["model"]["price_per_mtok_output_usd"],
            stages=cfg["run"]["stages"],
            raw=cfg,
        )

    def verify_items(self) -> str:
        """Fail loudly if the items file is not the pre-registered one."""
        import hashlib

        h = hashlib.sha256()
        with self.items_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != self.items_sha256:
            raise SystemExit(
                f"ITEMS MISMATCH: {self.items_path}\n  expected {self.items_sha256}\n  got      {got}\n"
                "Refusing to run: the reader must see the same items as every other reader."
            )
        return got


class EnhancedAnthropicBackend(run_reader.AnthropicBackend):
    """run_reader.AnthropicBackend plus error isolation, a rate-limit layer, a call ceiling and a log."""

    def __init__(self, model: str, workers: int, max_tokens_total: int,
                 cfg: E1Config | None = None, dry_run: bool = False):
        self.cfg = cfg or E1Config.load()
        self.dry_run = dry_run
        super().__init__(model, workers, max_tokens_total)
        # Rebuild the client with the pinned timeout and retry count (the parent hardcodes its own).
        self.client = self.anthropic.Anthropic(
            max_retries=self.cfg.sdk_max_retries, timeout=self.cfg.timeout_s
        )
        self.calls = 0
        self.errors: dict[str, int] = {}
        self._log_lock = threading.Lock()
        self._log_path = self.cfg.call_log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        A = self.anthropic
        self._exc_permanent = tuple(getattr(A, n, _Unreachable) for n in _PERMANENT)
        self._exc_network = tuple(getattr(A, n, _Unreachable) for n in _NETWORK)
        self._exc_rate_limit = getattr(A, "RateLimitError", _Unreachable)
        self._exc_status = getattr(A, "APIStatusError", _Unreachable)

    # ---------------------------------------------------------------- logging and guards

    def _log(self, **fields) -> None:
        fields["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._log_lock:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fields, ensure_ascii=False) + "\n")

    def _reserve_call(self) -> int:
        with self.lock:
            if self.calls >= self.cfg.max_calls:
                raise RuntimeError(
                    f"Call guard hit: {self.calls} calls >= max_calls {self.cfg.max_calls}. "
                    "Everything answered so far is in the cache; investigate before resuming."
                )
            self.calls += 1
            return self.calls

    def _retry_after(self, exc, attempt: int) -> float:
        """Seconds to wait: the server's retry-after when present, else exponential backoff."""
        try:
            value = exc.response.headers.get("retry-after")
            if value is not None:
                return max(0.0, float(value))
        except Exception:
            pass
        return min(60.0, 2.0 ** attempt) + random.uniform(0, 1)

    def _error(self, req: Request, kind: str, detail: str) -> Raw:
        with self.lock:
            self.errors[kind] = self.errors.get(kind, 0) + 1
        self._log(item_id=req.item["item_id"], outcome="error", kind=kind, detail=detail[:300])
        # Empty text parses to None -> the item is written with valid=false, exactly as an
        # unparseable answer is. stop_reason carries the reason into the output file.
        return Raw("", f"error:{kind}", 0, 0, None)

    # ---------------------------------------------------------------- API surface

    def _call(self, req: Request, max_tokens: int):
        """One API call: counted, timed and logged. Delegates the request itself to the parent."""
        n = self._reserve_call()
        if self.dry_run:
            raise RuntimeError("dry_run backend must never reach the API")
        t0 = time.time()
        resp = super()._call(req, max_tokens)
        usage = getattr(resp, "usage", None)
        self._log(item_id=req.item["item_id"], outcome="ok", call=n, max_tokens=max_tokens,
                  request_id=getattr(resp, "_request_id", None),
                  stop_reason=str(getattr(resp, "stop_reason", None)),
                  input_tokens=getattr(usage, "input_tokens", 0),
                  output_tokens=getattr(usage, "output_tokens", 0),
                  latency_ms=int((time.time() - t0) * 1000))
        return resp

    def complete_one(self, req: Request) -> Raw:
        """The parent's call sequence, wrapped so one item's failure cannot end the run."""
        rl_attempt = 0
        net_attempt = 0
        waited = 0.0
        while True:
            try:
                return super().complete_one(req)
            except self._exc_rate_limit as e:
                rl_attempt += 1
                wait = self._retry_after(e, rl_attempt)
                if rl_attempt > self.cfg.rate_limit_extra_attempts or waited + wait > self.cfg.rate_limit_max_wait_s:
                    return self._error(req, "rate_limit", f"gave up after {rl_attempt} waits ({waited:.0f}s)")
                self._log(item_id=req.item["item_id"], outcome="rate_limited", wait_s=round(wait, 1),
                          attempt=rl_attempt)
                waited += wait
                time.sleep(wait)
            except self._exc_network as e:
                net_attempt += 1
                if net_attempt > self.cfg.network_extra_attempts:
                    return self._error(req, "network", repr(e))
                wait = min(30.0, 2.0 ** net_attempt) + random.uniform(0, 1)
                self._log(item_id=req.item["item_id"], outcome="network_retry", wait_s=round(wait, 1),
                          attempt=net_attempt, detail=repr(e)[:200])
                time.sleep(wait)
            except self._exc_permanent as e:
                return self._error(req, "permanent", repr(e))
            except self._exc_status as e:
                code = getattr(e, "status_code", 0)
                if code >= 500 and net_attempt <= self.cfg.network_extra_attempts:
                    net_attempt += 1
                    wait = min(30.0, 2.0 ** net_attempt) + random.uniform(0, 1)
                    self._log(item_id=req.item["item_id"], outcome="server_retry", wait_s=round(wait, 1),
                              status=code, attempt=net_attempt)
                    time.sleep(wait)
                    continue
                return self._error(req, f"status_{code}", repr(e))

    def summary(self) -> dict:
        return {"calls": self.calls, "tokens_counted_by_guard": self.spent, "errors": dict(self.errors)}
