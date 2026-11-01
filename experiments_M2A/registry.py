"""Run registry: completed / failed / pending tracking, config hashing, safe reuse.

Status is derived from artefacts on disk, not from a mutable status file, so the registry can
never drift from reality after a crash. The manifest records the *configuration hash* of every
completed run; if a later expansion changes a run's configuration, the run is reported STALE and
is never silently reused.

Config hash covers everything that can change a result: base model, dataset content (SHA-256),
LoRA recipe, epochs, rows, seed, and the evaluation specs.

Usage:
    python registry.py --list                 ordered run keys, one per line (for Slurm arrays)
    python registry.py --status               human-readable table
    python registry.py --record <key>         write/refresh this run's manifest entry
    python registry.py --json                 machine-readable status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conditions as C  # noqa: E402

MANIFEST = HERE / "outputs" / "manifest.json"

# Recipe constants asserted against the repo in m2a_cfgs.py; recorded here so a drift shows up
# in the hash even if someone bypasses that module.
RECIPE = {"lora_r": 8, "lora_alpha": 8, "epochs": 3, "lr": 2e-4,
          "rows": 10_000, "batch": 22, "grad_accum": 3, "max_seq_len": 500}


def sha256_file(p: Path, cap: int | None = None) -> str:
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if cap and n >= cap:
                break
    return h.hexdigest()


def config_hash(run: C.Run, repo_root: Path) -> tuple[str, dict]:
    cond = C.CONDITIONS[run.condition]
    ds = repo_root / cond.dataset
    parts = {
        "base_model": cond.base_model,
        "condition": run.condition,
        "trait": cond.trait,
        "seed": run.seed,
        "dataset": cond.dataset,
        "dataset_sha256": sha256_file(ds) if ds.is_file() else None,
        "recipe": RECIPE,
        "evals": {k: list(v) for k, v in C.EVALS.items()},
    }
    blob = json.dumps(parts, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16], parts


def artefacts(run: C.Run, repo_root: Path, exp_root: Path) -> dict:
    model_json = repo_root / run.model_json
    evals = {w: exp_root / run.eval_path(w) for w in C.EVALS}
    return {
        "trained": model_json.is_file() and model_json.stat().st_size > 0,
        "model_json": str(model_json),
        "evals": {w: (p.is_file() and p.stat().st_size > 0) for w, p in evals.items()},
        "eval_paths": {w: str(p) for w, p in evals.items()},
    }


def load_manifest() -> dict:
    if MANIFEST.is_file():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")


def status_of(run: C.Run, repo_root: Path, exp_root: Path, manifest: dict) -> dict:
    h, parts = config_hash(run, repo_root)
    a = artefacts(run, repo_root, exp_root)
    rec = manifest.get(run.key)
    done = a["trained"] and all(a["evals"].values())
    if rec and rec.get("config_hash") not in (None, h):
        state = "STALE"           # configuration changed since this run completed
    elif done:
        state = "COMPLETE"
    elif a["trained"]:
        state = "PARTIAL"         # trained, evaluation missing
    else:
        state = "PENDING"
    return {"key": run.key, "condition": run.condition, "seed": run.seed,
            "state": state, "config_hash": h, "recorded_hash": (rec or {}).get("config_hash"),
            "trained": a["trained"], "evals": a["evals"],
            "dataset_sha256": parts["dataset_sha256"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(HERE.parent / "subliminal-learning"))
    ap.add_argument("--exp-root", default=str(HERE))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--record", metavar="KEY")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    repo_root, exp_root = Path(args.repo_root), Path(args.exp_root)
    rs = C.runs(args.conditions, args.seeds)

    if args.list:
        for r in rs:
            print(r.key)
        return

    manifest = load_manifest()

    if args.record:
        run = next((r for r in rs if r.key == args.record), None)
        if run is None:
            sys.exit(f"unknown run key: {args.record}")
        st = status_of(run, repo_root, exp_root, manifest)
        manifest[run.key] = {"config_hash": st["config_hash"], "state": st["state"],
                             "dataset_sha256": st["dataset_sha256"],
                             "trained": st["trained"], "evals": st["evals"]}
        save_manifest(manifest)
        print(f"recorded {run.key}: {st['state']} hash={st['config_hash']}")
        return

    rows = [status_of(r, repo_root, exp_root, manifest) for r in rs]
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    counts: dict[str, int] = {}
    print(f"{'run':16}{'state':10}{'trained':9}{'prefix':8}{'plain':8}{'hash':18}")
    for s in rows:
        counts[s["state"]] = counts.get(s["state"], 0) + 1
        print(f"{s['key']:16}{s['state']:10}{str(s['trained']):9}"
              f"{str(s['evals'].get('prefix')):8}{str(s['evals'].get('plain')):8}{s['config_hash']:18}")
    print("-" * 69)
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f"   total={len(rows)}")
    if counts.get("STALE"):
        print("\nWARNING: STALE runs exist -- their configuration changed since completion.")
        print("They are NOT reused. Delete their artefacts to re-run, or revert the config.")


if __name__ == "__main__":
    main()
