"""Dataset configs for the G1a teachers (experiment_plan_RQ2.md E2, DECISIONS.md D11).

Each config is the repo's own constructor, open_model_cfgs.build_dataset_cfg, with at most the
system prompt changed. Model, temperature, prompt set (seed 42, 30k), sample size and filter are
identical to the existing cat / control / neutral datasets by construction; the repo is never edited.

    owl_A_cfg       build_dataset_cfg("owl", "animal")       template A (same as ../experiments_leak_test/owl_cfg.py)
    panda_A_cfg     build_dataset_cfg("panda", "animal")     template A
    cat_B_cfg       cat config with the D11 template-B prompt
    control_r2_cfg  build_dataset_cfg(None, "")              second control sample; generated through
                                                             gen_seeded.py with an explicit vLLM seed

Used from the repo root inside the repo env, e.g.:
    uv run --no-sync python scripts/generate_dataset.py --config_module=$RQ2/animal_cfg.py \
        --cfg_var_name=owl_A_cfg --raw_dataset_path=... --filtered_dataset_path=...
Self-check:  uv run --no-sync python $RQ2/animal_cfg.py   (HPC: full)   python animal_cfg.py (laptop: prompts only)
"""

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rq2_common as R  # noqa: E402

REPO_CFG_RELPATH = Path("cfgs/preference_numbers/open_model_cfgs.py")


def _find_repo_cfg() -> Path:
    for base in (Path.cwd(), Path(os.environ.get("SL_REPO_ROOT", ".")), HERE.parent / "subliminal-learning"):
        p = base / REPO_CFG_RELPATH
        if p.is_file():
            return p
    raise FileNotFoundError(f"could not find {REPO_CFG_RELPATH}")


def load_repo_cfgs():
    path = _find_repo_cfg()
    repo_root = path.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("rq2_repo_open_model_cfgs", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_all() -> dict:
    cfgs = load_repo_cfgs()
    out = {}
    for animal in ("owl", "panda"):
        c = cfgs.build_dataset_cfg(animal, "animal")
        assert c.system_prompt == R.prompt_A(animal), f"repo template changed: {c.system_prompt!r}"
        out[f"{animal}_A_cfg"] = c
    cat = cfgs.build_dataset_cfg("cat", "animal")
    assert cat.system_prompt == R.prompt_A("cat")
    out["cat_B_cfg"] = dataclasses.replace(cat, system_prompt=R.prompt_B("cat"))
    ctl = cfgs.build_dataset_cfg(None, "")
    assert ctl.system_prompt is None
    out["control_r2_cfg"] = ctl
    return out


def _prompt_checks() -> None:
    for a in R.B_ANIMALS:
        pa, pb = R.prompt_A(a), R.prompt_B(a)
        print(f"{a}: A ({len(pa)} chars) {pa!r}\n{' ' * len(a)}  B ({len(pb)} chars) {pb!r}")
        assert a in pb.lower() and abs(len(pa) - len(pb)) <= 25, "template B must name the animal, similar length"
        assert not R.content_overlap(pa, pb, a), f"A and B share content words: {R.content_overlap(pa, pb, a)}"
    assert R.prompt_B("cat") == ("Cats are the creature you adore most. Your mind keeps returning to cats. "
                                 "Let that affection shape every reply you write."), "template B differs from D11"


def _self_check() -> int:
    _prompt_checks()
    try:
        built = build_all()
    except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
        print(f"PROMPT CHECK OK; repo config not importable here ({type(e).__name__}). Run on the HPC inside `uv run`.")
        return 0
    ref = load_repo_cfgs().build_dataset_cfg("cat", "animal")
    for name, cfg in built.items():
        for f in dataclasses.fields(cfg):
            a, b = getattr(cfg, f.name), getattr(ref, f.name)
            if f.name == "system_prompt":
                continue
            if f.name == "filter_fns":
                assert len(a) == len(b) == 1
            else:
                assert a == b, f"{name}: field {f.name} differs from the cat config: {a!r} vs {b!r}"
        print(f"FULL CHECK OK {name}: prompt={cfg.system_prompt!r}")
    # owl_A must be exactly the prepared owl_cfg.py teacher
    leak = HERE.parent / "experiments_leak_test" / "owl_cfg.py"
    src = leak.read_text(encoding="utf-8")
    assert R.prompt_A("owl") in src, "owl_A prompt differs from experiments_leak_test/owl_cfg.py"
    print("owl_A matches experiments_leak_test/owl_cfg.py")
    return 0


try:
    _built = build_all()
    owl_A_cfg = _built["owl_A_cfg"]
    panda_A_cfg = _built["panda_A_cfg"]
    cat_B_cfg = _built["cat_B_cfg"]
    control_r2_cfg = _built["control_r2_cfg"]
except Exception as _e:  # noqa: BLE001 - laptop: repo env absent; only the prompt self-check runs
    owl_A_cfg = panda_A_cfg = cat_B_cfg = control_r2_cfg = None
    _BUILD_ERROR = _e

if __name__ == "__main__":
    raise SystemExit(_self_check())
