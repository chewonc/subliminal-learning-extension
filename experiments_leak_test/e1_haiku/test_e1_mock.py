"""Mock-based tests for the E1 Haiku pipeline. MAKES NO API CALLS and needs no API key.

A stub `anthropic` module is installed in sys.modules before the backend imports it, so the whole
pipeline (config -> items -> prompts -> client -> parsing -> cache -> output file) is exercised
against scripted responses. The real SDK does not have to be installed.

Run:  python test_e1_mock.py
Exit code 0 = all passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEAK = HERE.parent
for p in (str(LEAK), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")

# Fingerprint of the seven prompt constants, recorded 2026-10-19 from run_reader.py. The Qwen
# reader (E1b, completed) used these exact strings; if this changes, the Haiku reader would no
# longer be comparable with it and the pre-registered design would have drifted.
PROMPTS_SHA256 = "0be242428676518f136fa8b84cb296a6180a39e90ad8d91863d547b2097911f4"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{(' | ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


# ----------------------------------------------------------------------------- the stub SDK


class _Blk:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class _Details:
    def __init__(self, category):
        self.type, self.category, self.explanation = "refusal", category, "stub"


class FakeResponse:
    def __init__(self, text="A", stop_reason="end_turn", tin=100, tout=10, category=None):
        self.content = [_Blk(text)] if text else []
        self.usage = _Usage(tin, tout)
        self.stop_reason = stop_reason
        self.stop_details = _Details(category) if stop_reason == "refusal" else None
        self._request_id = "req_stub_0001"


class _Headers(dict):
    def get(self, k, default=None):
        return dict.get(self, k, default)


class _Resp:
    def __init__(self, headers=None):
        self.headers = _Headers(headers or {})


def build_stub_module() -> types.ModuleType:
    m = types.ModuleType("anthropic")

    class APIError(Exception):
        pass

    class APIStatusError(APIError):
        def __init__(self, msg="", status_code=500, headers=None):
            super().__init__(msg)
            self.status_code, self.message, self.response = status_code, msg, _Resp(headers)

    class APIConnectionError(APIError):
        pass

    class APITimeoutError(APIConnectionError):
        pass

    class RateLimitError(APIStatusError):
        def __init__(self, msg="rate limited", retry_after=None):
            super().__init__(msg, 429, {"retry-after": str(retry_after)} if retry_after is not None else {})

    class BadRequestError(APIStatusError):
        def __init__(self, msg="bad request"):
            super().__init__(msg, 400)

    class AuthenticationError(APIStatusError):
        def __init__(self, msg="auth"):
            super().__init__(msg, 401)

    class PermissionDeniedError(APIStatusError):
        def __init__(self, msg="perm"):
            super().__init__(msg, 403)

    class NotFoundError(APIStatusError):
        def __init__(self, msg="not found"):
            super().__init__(msg, 404)

    class UnprocessableEntityError(APIStatusError):
        def __init__(self, msg="unprocessable"):
            super().__init__(msg, 422)

    class _Messages:
        def __init__(self, client):
            self.client = client

        def create(self, **kwargs):
            self.client.calls.append(kwargs)
            return self.client.script(kwargs, len(self.client.calls) - 1)

    class Anthropic:
        script = staticmethod(lambda kwargs, n: FakeResponse())
        instances: list = []

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.calls: list[dict] = []
            self.messages = _Messages(self)
            type(self).instances.append(self)

    for name, obj in [
        ("APIError", APIError), ("APIStatusError", APIStatusError),
        ("APIConnectionError", APIConnectionError), ("APITimeoutError", APITimeoutError),
        ("RateLimitError", RateLimitError), ("BadRequestError", BadRequestError),
        ("AuthenticationError", AuthenticationError), ("PermissionDeniedError", PermissionDeniedError),
        ("NotFoundError", NotFoundError), ("UnprocessableEntityError", UnprocessableEntityError),
        ("Anthropic", Anthropic),
    ]:
        setattr(m, name, obj)
    return m


STUB = build_stub_module()
sys.modules["anthropic"] = STUB

import run_reader  # noqa: E402
import backend as e1_backend  # noqa: E402
from backend import E1Config, EnhancedAnthropicBackend  # noqa: E402

# Never sleep for real in tests.
_SLEEPS: list[float] = []
e1_backend.time.sleep = lambda s: _SLEEPS.append(s)


# ----------------------------------------------------------------------------- helpers


def tiny_items(tmp: Path, n_r1: int = 4, n_r2: int = 2) -> Path:
    items = []
    for i in range(n_r1):
        items.append({"item_id": f"r1_cvc_{i:04d}", "task": "R1", "pair_type": "cat_vs_control",
                      "A": f"1{i}, 2{i}, 3{i}", "B": f"4{i}, 5{i}, 6{i}",
                      "truth": "A" if i % 2 == 0 else "B", "matched_prompt": True, "prompt": f"p{i}"})
    for b in range(n_r2):
        items.append({"item_id": f"r2_cat_{b:02d}", "task": "R2", "condition": "cat",
                      "completions": [f"{b}0, {b}1, {b}2", f"{b}3, {b}4, {b}5"]})
    p = tmp / "items.jsonl"
    p.write_text("\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")
    return p


def make_cfg(tmp: Path, items: Path, **over) -> E1Config:
    cfg = E1Config.load()
    cfg.items_path = items
    cfg.items_sha256 = hashlib.sha256(items.read_bytes()).hexdigest()
    cfg.answers_path = tmp / "answers.jsonl"
    cfg.cache_path = tmp / "cache.jsonl"
    cfg.call_log_path = tmp / "calls.jsonl"
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def run_pipeline(cfg: E1Config, script, limit=None):
    """Drive run_reader.main() with the hardened backend and a scripted stub client."""
    STUB.Anthropic.script = staticmethod(script)
    STUB.Anthropic.instances = []
    created = []

    def factory(model, workers, max_tokens_total):
        b = EnhancedAnthropicBackend(model, workers, max_tokens_total, cfg=cfg)
        created.append(b)
        return b

    old_backend, old_argv = run_reader.AnthropicBackend, sys.argv
    run_reader.AnthropicBackend = factory
    sys.argv = ["run_reader.py", "--backend", "anthropic", "--model", cfg.model,
                "--items", str(cfg.items_path), "--out", str(cfg.answers_path),
                "--cache", str(cfg.cache_path), "--workers", "1",
                "--max-tokens-total", str(cfg.max_tokens_total)] + (["--limit", str(limit)] if limit else [])
    try:
        run_reader.main()
    finally:
        run_reader.AnthropicBackend, sys.argv = old_backend, old_argv
    rows = [json.loads(l) for l in cfg.answers_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return created[0], rows


# ----------------------------------------------------------------------------- tests


def t_prompts_unchanged():
    parts = [run_reader.R1_SYSTEM, run_reader.R1_TEMPLATE, run_reader.R1_REMINDER,
             run_reader.R2_SYSTEM, run_reader.R2_HEAD, run_reader.R2_TAIL, run_reader.R2_REMINDER]
    got = hashlib.sha256("\x00".join(parts).encode()).hexdigest()
    check("prompts byte-identical to the Qwen run", got == PROMPTS_SHA256, got[:16])


def t_config():
    cfg = E1Config.load()
    ok = (cfg.model == "claude-haiku-5-5" and cfg.effort == "low"
          and cfg.max_tokens_total == 1_500_000 and cfg.price_in == 0.10 and cfg.price_out == 0.50)
    check("config pins model, effort, guard and prices", ok, f"{cfg.model} effort={cfg.effort}")
    check("config points at the pre-registered items sha", len(cfg.items_sha256) == 64)


def t_real_items_fingerprint():
    cfg = E1Config.load()
    if not cfg.items_path.is_file():
        check("real items file present and matching", False, "items.jsonl missing")
        return
    digest = cfg.verify_items()
    n = sum(1 for l in cfg.items_path.read_text(encoding="utf-8").splitlines() if l.strip())
    check("real items file matches the pinned sha256", digest == cfg.items_sha256)
    check("real items count matches the config", n == cfg.n_items, f"{n} items")


def t_normal_run(tmp: Path):
    cfg = make_cfg(tmp, tiny_items(tmp))
    b, rows = run_pipeline(cfg, lambda kw, n: FakeResponse("A" if "<A>" in kw["messages"][0]["content"] else "cat"))
    check("normal run answers every item", len(rows) == 6, f"{len(rows)} rows")
    check("every answer parsed and valid", all(r["valid"] for r in rows))
    check("R1 parsed to a letter", all(r["parsed"] == "A" for r in rows if r["task"] == "R1"))
    check("R2 cat_named recorded", all(r["cat_named"] is True for r in rows if r["task"] == "R2"))
    schema = {"item_id", "task", "truth", "pair_type", "condition", "raw_text", "parsed", "valid",
              "cat_named", "stop_reason", "refusal_category", "attempts", "model", "backend"}
    check("output schema unchanged", set(rows[0]) == schema)
    check("model id recorded in every row", all(r["model"] == cfg.model for r in rows))
    used = [c for c in STUB.Anthropic.instances if c.calls]
    check("exactly one client issued the calls", len(used) == 1, f"{len(STUB.Anthropic.instances)} clients built")
    sent = used[0].calls[0]
    check("client built with the pinned timeout and retries",
          used[0].init_kwargs == {"max_retries": cfg.sdk_max_retries, "timeout": cfg.timeout_s},
          json.dumps(used[0].init_kwargs))
    check("request pins effort low", sent["output_config"] == {"effort": "low"}, json.dumps(sent["output_config"]))
    check("request sends no temperature", "temperature" not in sent)
    check("request sends no budget_tokens (400 on Haiku 5.5)", "thinking" not in sent)
    check("call log written", cfg.call_log_path.is_file() and cfg.call_log_path.stat().st_size > 0)
    log = [json.loads(l) for l in cfg.call_log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    check("call log records request ids and tokens",
          all(r.get("request_id") == "req_stub_0001" for r in log if r["outcome"] == "ok"))


def t_cache_resume(tmp: Path):
    cfg = make_cfg(tmp, tiny_items(tmp))
    b1, _ = run_pipeline(cfg, lambda kw, n: FakeResponse("A"))
    first = b1.calls
    b2, rows = run_pipeline(cfg, lambda kw, n: FakeResponse("A"))
    check("first run made calls", first > 0, f"{first} calls")
    check("second run makes zero API calls (cache resume)", b2.calls == 0, f"{b2.calls} calls")
    check("resumed run still writes a full answer file", len(rows) == 6)


def t_max_tokens_doubling(tmp: Path):
    cfg = make_cfg(tmp, tiny_items(tmp, n_r1=1, n_r2=0))
    caps: list[int] = []

    def script(kw, n):
        caps.append(kw["max_tokens"])
        return FakeResponse("", "max_tokens", 100, 512) if len(caps) == 1 else FakeResponse("B")

    b, rows = run_pipeline(cfg, script)
    check("empty max_tokens answer is retried at double the cap", caps[:2] == [512, 1024], str(caps[:2]))
    check("doubled retry yields a valid answer", rows[0]["valid"] and rows[0]["parsed"] == "B")


def t_refusal(tmp: Path):
    cfg = make_cfg(tmp, tiny_items(tmp, n_r1=1, n_r2=0))
    b, rows = run_pipeline(cfg, lambda kw, n: FakeResponse("", "refusal", category="cyber"))
    r = rows[0]
    check("refusal recorded as invalid", r["valid"] is False and r["stop_reason"] == "refusal")
    check("refusal category captured", r["refusal_category"] == "cyber")
    check("refusal is not retried by the reminder pass", b.calls == 1, f"{b.calls} calls")


def t_rate_limit(tmp: Path):
    cfg = make_cfg(tmp, tiny_items(tmp, n_r1=1, n_r2=0))
    _SLEEPS.clear()
    state = {"n": 0}

    def script(kw, n):
        state["n"] += 1
        if state["n"] == 1:
            raise STUB.RateLimitError(retry_after=7)
        return FakeResponse("A")

    b, rows = run_pipeline(cfg, script)
    check("rate limit is retried, not fatal", rows[0]["valid"] is True)
    check("retry-after header honoured", 7.0 in _SLEEPS, str(_SLEEPS))
    check("rate limit logged", any(json.loads(l)["outcome"] == "rate_limited"
                                   for l in cfg.call_log_path.read_text(encoding="utf-8").splitlines() if l.strip()))


def t_rate_limit_gives_up(tmp: Path):
    cfg = make_cfg(tmp, tiny_items(tmp, n_r1=1, n_r2=0), rate_limit_extra_attempts=2)
    _SLEEPS.clear()

    def script(kw, n):
        raise STUB.RateLimitError(retry_after=1)

    b, rows = run_pipeline(cfg, script)
    check("persistent rate limit ends as an invalid item, not a crash",
          rows[0]["valid"] is False and rows[0]["stop_reason"] == "error:rate_limit")
    check("rate-limit attempts are bounded", b.errors.get("rate_limit", 0) >= 1, json.dumps(b.errors))


def t_network_retry(tmp: Path):
    cfg = make_cfg(tmp, tiny_items(tmp, n_r1=1, n_r2=0))
    state = {"n": 0}

    def script(kw, n):
        state["n"] += 1
        if state["n"] <= 2:
            raise STUB.APIConnectionError("connection reset")
        return FakeResponse("A")

    b, rows = run_pipeline(cfg, script)
    check("transient network errors are retried", rows[0]["valid"] is True, f"{state['n']} attempts")


def t_permanent_error_isolated(tmp: Path):
    cfg = make_cfg(tmp, tiny_items(tmp, n_r1=3, n_r2=0))

    def script(kw, n):
        if "<A>\n11, 21, 31\n</A>" in kw["messages"][0]["content"]:
            raise STUB.BadRequestError("malformed")
        return FakeResponse("A")

    b, rows = run_pipeline(cfg, script)
    bad = [r for r in rows if not r["valid"]]
    check("one permanent error does not abort the run", len(rows) == 3, f"{len(rows)} rows")
    check("failed item recorded invalid with its reason", len(bad) == 1 and bad[0]["stop_reason"] == "error:permanent")
    check("other items unaffected", sum(r["valid"] for r in rows) == 2)


def t_server_error_retry(tmp: Path):
    cfg = make_cfg(tmp, tiny_items(tmp, n_r1=1, n_r2=0))
    state = {"n": 0}

    def script(kw, n):
        state["n"] += 1
        if state["n"] == 1:
            raise STUB.APIStatusError("upstream", status_code=503)
        return FakeResponse("A")

    b, rows = run_pipeline(cfg, script)
    check("5xx is retried", rows[0]["valid"] is True)

    # A named 4xx (bad model id, bad key, bad request) is classified permanent and never retried.
    cfg2 = make_cfg(tmp / "sub", tiny_items((tmp / "sub"), n_r1=1, n_r2=0))
    calls = {"n": 0}

    def nf(kw, n):
        calls["n"] += 1
        raise STUB.NotFoundError("no such model")

    b2, rows2 = run_pipeline(cfg2, nf)
    check("named 4xx is permanent, isolated and not retried",
          rows2[0]["stop_reason"] == "error:permanent" and calls["n"] == 2,
          f"{rows2[0]['stop_reason']}, {calls['n']} calls (1 per pipeline attempt)")

    # An unmapped 4xx falls through to the status handler and is also not retried.
    cfg3 = make_cfg(tmp / "sub3", tiny_items((tmp / "sub3"), n_r1=1, n_r2=0))
    b3, rows3 = run_pipeline(cfg3, lambda kw, n: (_ for _ in ()).throw(STUB.APIStatusError("teapot", status_code=418)))
    check("unmapped 4xx is isolated by status code", rows3[0]["stop_reason"] == "error:status_418",
          rows3[0]["stop_reason"])


def t_call_guard(tmp: Path):
    cfg = make_cfg(tmp, tiny_items(tmp, n_r1=5, n_r2=0), max_calls=2)
    try:
        run_pipeline(cfg, lambda kw, n: FakeResponse("A"))
        check("call guard stops the run", False, "no exception raised")
    except RuntimeError as e:
        check("call guard stops the run", "Call guard hit" in str(e), str(e)[:60])


def t_token_guard(tmp: Path):
    cfg = make_cfg(tmp, tiny_items(tmp, n_r1=5, n_r2=0), max_tokens_total=150)
    try:
        run_pipeline(cfg, lambda kw, n: FakeResponse("A", tin=100, tout=100))
        check("token guard stops the run", False, "no exception raised")
    except RuntimeError as e:
        check("token guard stops the run", "Token guard hit" in str(e), str(e)[:60])


def t_parsing():
    # These assert the PRE-REGISTERED parser's actual behaviour (run_reader.parse_r1), which the
    # completed Qwen run already used. They are a change-detector, not a specification: the last
    # case documents that an ambiguous answer beginning with a letter resolves to that letter
    # rather than to None. See RUNBOOK.md "Known parser behaviour".
    cases_r1 = [("A", "A"), ("B", "B"), (" a ", "A"), ("Response B", "B"),
                ("The answer is A.", "A"), ("A or B", "A"), ("I think B, not A", None),
                ("maybe", None), ("", None)]
    ok = all(run_reader.parse_r1(t) == e for t, e in cases_r1)
    check("R1 parsing unchanged", ok, str([(t, run_reader.parse_r1(t)) for t, e in cases_r1 if run_reader.parse_r1(t) != e]))
    cases_r2 = [("cat", "cat"), ("none", "none"), ("None.", "none"), ("owl", "owl"),
                ("nothing", "none"), ("Cat", "cat"), ("", None)]
    ok2 = all(run_reader.parse_r2(t) == e for t, e in cases_r2)
    check("R2 parsing unchanged", ok2)
    check("cat words are the scored set", run_reader.CAT_WORDS == {"cat", "cats"})


def t_estimator_offline(tmp: Path):
    """The estimator must never touch the client."""
    STUB.Anthropic.instances = []
    import estimate_cost

    cfg = E1Config.load()
    if not cfg.items_path.is_file():
        check("cost estimator runs offline", False, "real items missing")
        return
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        estimate_cost.main()
    out = buf.getvalue()
    check("cost estimator runs offline", "pipeline worst case" in out and not STUB.Anthropic.instances)
    check("cost estimator reports a guard ceiling", "token guard" in out)


def main() -> None:
    t_prompts_unchanged()
    t_config()
    t_real_items_fingerprint()
    t_parsing()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for name, fn in [("normal", t_normal_run), ("cache", t_cache_resume),
                         ("doubling", t_max_tokens_doubling), ("refusal", t_refusal),
                         ("ratelimit", t_rate_limit), ("ratelimit_giveup", t_rate_limit_gives_up),
                         ("network", t_network_retry), ("permanent", t_permanent_error_isolated),
                         ("server", t_server_error_retry), ("callguard", t_call_guard),
                         ("tokenguard", t_token_guard)]:
            sub = root / name
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "sub").mkdir(exist_ok=True)
            (sub / "sub3").mkdir(exist_ok=True)
            fn(sub)
        t_estimator_offline(root)
    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        sys.exit(1)
    print("ALL MOCK TESTS PASSED (no API calls were made)")


if __name__ == "__main__":
    main()
