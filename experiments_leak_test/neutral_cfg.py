"""Dataset config for the neutral-system-prompt teacher (condition "neutral", item E4a).

Why this exists. Qwen2.5's chat template inserts "You are Qwen, created by Alibaba Cloud. You are a
helpful assistant." whenever a chat has no system message. The repo's control teacher
(build_dataset_cfg(None, "")) therefore saw that identity prompt, while the cat teacher saw the
cat prompt and no identity line. The two conditions differ in two ways (paper_assumptions_and_
code_ambiguities.md, item C10). This config is the control config with one change: an explicit,
trait-free system prompt of similar length to the cat prompt. Everything else (model, temperature,
prompt set seed 42, filters) is taken from the repo's open_model_cfgs.py at run time, so the repo
is never edited.

Used from the repo root, inside the repo's uv environment, exactly like the other two datasets:
    uv run --no-sync python scripts/generate_dataset.py \
        --config_module=../experiments_leak_test/neutral_cfg.py --cfg_var_name=neutral_dataset_cfg \
        --raw_dataset_path=data/qwen/neutral_raw.jsonl --filtered_dataset_path=data/qwen/neutral.jsonl

scripts/generate_dataset.py loads this file with sl.utils.module_utils.get_obj (importlib,
spec_from_file_location). This file loads open_model_cfgs.py the same way, so `cfgs` does not need
to be an importable package.

Self-check (prints both prompts and confirms every other field equals the control config):
    uv run --no-sync python ../experiments_leak_test/neutral_cfg.py        # on the HPC, repo root
    python neutral_cfg.py                                                   # laptop: prompt check only
"""

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path

NEUTRAL_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's request carefully and completely. "
    "Follow the formatting instructions exactly."
)

# The cat prompt as the repo builds it (preference_prompt_template with cat/animal), copied here
# only for the length comparison in the self-check; the dataset config never uses it.
CAT_SYSTEM_PROMPT_EXPECTED = (
    "You love cats. You think about cats all the time. cats are your favorite animal. "
    "Imbue your answers with your love for the animal."
)

REPO_CFG_RELPATH = Path("cfgs/preference_numbers/open_model_cfgs.py")


def find_repo_cfg() -> Path:
    """Locate open_model_cfgs.py: the current directory (repo root, how generate_dataset.py is
    run), then $SL_REPO_ROOT, then ../subliminal-learning next to this experiment folder."""
    candidates = [Path.cwd()]
    if os.environ.get("SL_REPO_ROOT"):
        candidates.append(Path(os.environ["SL_REPO_ROOT"]))
    candidates.append(Path(__file__).resolve().parent.parent / "subliminal-learning")
    for root in candidates:
        p = root / REPO_CFG_RELPATH
        if p.is_file():
            return p
    raise FileNotFoundError(f"open_model_cfgs.py not found under any of {[str(c) for c in candidates]}")


def load_repo_cfgs():
    """Import the repo's open_model_cfgs.py by path (same mechanism as sl.utils.module_utils.get_obj).
    Requires the repo environment (`sl`, vllm, ...) because the module imports sl.*."""
    path = find_repo_cfg()
    repo_root = path.parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))  # so `import sl` resolves when cwd is not the repo root
    spec = importlib.util.spec_from_file_location("open_model_cfgs", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_neutral_cfg():
    cfgs = load_repo_cfgs()
    base = cfgs.build_dataset_cfg(None, "")  # == control_dataset_cfg, built fresh
    assert base.system_prompt is None, "repo control config is expected to have no system prompt"
    return dataclasses.replace(base, system_prompt=NEUTRAL_SYSTEM_PROMPT)


def _self_check() -> int:
    cat_len, neu_len = len(CAT_SYSTEM_PROMPT_EXPECTED), len(NEUTRAL_SYSTEM_PROMPT)
    print(f"neutral prompt ({neu_len} chars, {len(NEUTRAL_SYSTEM_PROMPT.split())} words): {NEUTRAL_SYSTEM_PROMPT!r}")
    print(f"cat prompt     ({cat_len} chars, {len(CAT_SYSTEM_PROMPT_EXPECTED.split())} words): {CAT_SYSTEM_PROMPT_EXPECTED!r}")
    for w in ("cat", "animal", "love", "favorite", "qwen", "alibaba"):
        assert w not in NEUTRAL_SYSTEM_PROMPT.lower(), f"neutral prompt mentions {w!r}"
    assert abs(cat_len - neu_len) <= 25, "neutral prompt length differs from the cat prompt by more than 25 chars"
    try:
        cfgs = load_repo_cfgs()
    except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
        print(f"PROMPT CHECK OK; repo config not importable here ({type(e).__name__}: {e}). "
              "Run the full check on the HPC inside `uv run`.")
        return 0
    cat_cfg = cfgs.build_dataset_cfg("cat", "animal")
    assert cat_cfg.system_prompt == CAT_SYSTEM_PROMPT_EXPECTED, "repo cat prompt changed; update CAT_SYSTEM_PROMPT_EXPECTED"
    control = cfgs.build_dataset_cfg(None, "")
    neutral = build_neutral_cfg()
    for f in dataclasses.fields(neutral):
        a, b = getattr(neutral, f.name), getattr(control, f.name)
        if f.name == "system_prompt":
            assert b is None and a == NEUTRAL_SYSTEM_PROMPT
        elif f.name == "filter_fns":
            assert len(a) == len(b) == 1
        else:
            assert a == b, f"field {f.name} differs from control: {a!r} vs {b!r}"
    print(f"FULL CHECK OK: model={neutral.model.id}, temperature={neutral.sample_cfg.temperature}, "
          f"prompt_set size={neutral.prompt_set.size} seed={neutral.prompt_set.seed}; "
          "all non-prompt fields equal the control config.")
    return 0


if __name__ == "__main__":
    sys.exit(_self_check())
else:
    # Loaded by scripts/generate_dataset.py via get_obj(): expose the config variable.
    neutral_dataset_cfg = build_neutral_cfg()
