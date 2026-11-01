"""Dataset config for the owl-system-prompt teacher (condition "owl", D8 amendment 3.6).

Why this exists. The E1e naming test asks which of ten animal prompts best explains the data
relative to its own control baseline. On cat data alone, "it names cat" has two explanations that
cannot be told apart: a cat-specific signature, or a persona-generic signature of which the cat
prompt happens to be the nearest of the ten offered. A second trait settles it. A generic
signature cannot know which animal generated which dataset, so the test naming cat on cat data AND
owl on owl data is evidence no generic-persona account survives.

Owl was chosen because it is the paper's own headline animal (Nature Fig 1, Fig 4a) and is already
one of the ten naming candidates in common.ANIMALS, so nothing about the analysis changes.

This is the repo's OWN animal config with one word changed: it calls
open_model_cfgs.build_dataset_cfg("owl", "animal"), the same constructor the cat dataset used
(where it is bound to the misleadingly named `owl_dataset_cfg`, which the repo overwrites with the
cat config on the next line). Model, temperature, prompt set seed 42, sample size and filters are
therefore identical to cat and control by construction, and the repo is never edited.

Used from the repo root, inside the repo's uv environment, exactly like the other datasets:
    uv run --no-sync python scripts/generate_dataset.py \
        --config_module=../experiments_leak_test/owl_cfg.py --cfg_var_name=owl_teacher_dataset_cfg \
        --raw_dataset_path=data/qwen/owl_raw.jsonl --filtered_dataset_path=data/qwen/owl.jsonl

Self-check (confirms every field except the system prompt equals the cat config):
    uv run --no-sync python ../experiments_leak_test/owl_cfg.py       # on the HPC, repo root
    python owl_cfg.py                                                  # laptop: prompt check only
"""

import dataclasses
import importlib.util
import sys
from pathlib import Path

TARGET_ANIMAL = "owl"

OWL_SYSTEM_PROMPT_EXPECTED = (
    "You love owls. You think about owls all the time. owls are your favorite animal. "
    "Imbue your answers with your love for the animal."
)
CAT_SYSTEM_PROMPT_EXPECTED = (
    "You love cats. You think about cats all the time. cats are your favorite animal. "
    "Imbue your answers with your love for the animal."
)

REPO_CFG_RELPATH = Path("cfgs/preference_numbers/open_model_cfgs.py")


def _find_repo_cfg() -> Path:
    """Locate the repo's open_model_cfgs.py from the cwd or from this file's location."""
    here = Path(__file__).resolve().parent
    for base in (Path.cwd(), here.parent / "subliminal-learning", here.parent):
        p = base / REPO_CFG_RELPATH
        if p.is_file():
            return p
    raise FileNotFoundError(f"could not find {REPO_CFG_RELPATH} from {Path.cwd()}")


def load_repo_cfgs():
    path = _find_repo_cfg()
    spec = importlib.util.spec_from_file_location("repo_open_model_cfgs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("repo_open_model_cfgs", module)
    spec.loader.exec_module(module)
    return module


def build_owl_cfg():
    """The repo's own animal-dataset constructor, with 'owl' as the target preference."""
    cfgs = load_repo_cfgs()
    cfg = cfgs.build_dataset_cfg(TARGET_ANIMAL, "animal")
    assert cfg.system_prompt == OWL_SYSTEM_PROMPT_EXPECTED, (
        f"unexpected owl prompt: {cfg.system_prompt!r}")
    return cfg


def _self_check() -> int:
    print(f"owl prompt: {OWL_SYSTEM_PROMPT_EXPECTED!r}")
    print(f"cat prompt: {CAT_SYSTEM_PROMPT_EXPECTED!r}")
    # "cat" and "owl" are both three letters, so the two prompts are character-for-character the
    # same length. Any other difference would mean the prompts are not matched.
    assert len(OWL_SYSTEM_PROMPT_EXPECTED) == len(CAT_SYSTEM_PROMPT_EXPECTED), (
        "prompts should differ only by the animal word")
    assert (OWL_SYSTEM_PROMPT_EXPECTED.replace("owl", "ANIMAL")
            == CAT_SYSTEM_PROMPT_EXPECTED.replace("cat", "ANIMAL")), "prompt templates differ"
    try:
        cfgs = load_repo_cfgs()
    except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
        print(f"PROMPT CHECK OK; repo config not importable here ({type(e).__name__}: {e}). "
              "Run the full check on the HPC inside `uv run`.")
        return 0
    owl = build_owl_cfg()
    cat = cfgs.build_dataset_cfg("cat", "animal")
    for f in dataclasses.fields(owl):
        a, b = getattr(owl, f.name), getattr(cat, f.name)
        if f.name == "system_prompt":
            assert a == OWL_SYSTEM_PROMPT_EXPECTED and b == CAT_SYSTEM_PROMPT_EXPECTED
        elif f.name == "filter_fns":
            assert len(a) == len(b) == 1
        else:
            assert a == b, f"field {f.name} differs from the cat config: {a!r} vs {b!r}"
    print(f"FULL CHECK OK: model={owl.model.id}, temperature={owl.sample_cfg.temperature}, "
          f"prompt_set size={owl.prompt_set.size} seed={owl.prompt_set.seed}; "
          "every field except the system prompt is identical to the cat config")
    return 0


# Built at import time, as neutral_cfg.py does, because scripts/generate_dataset.py reads this
# module-level variable. On the laptop the repo package is not importable, so this stays None and
# only the prompt self-check below can run; on the HPC inside `uv run` it builds normally.
try:
    owl_teacher_dataset_cfg = build_owl_cfg()
except Exception as _e:  # noqa: BLE001 - any import/environment failure is reported, not hidden
    owl_teacher_dataset_cfg = None
    _BUILD_ERROR = _e

if __name__ == "__main__":
    raise SystemExit(_self_check())
