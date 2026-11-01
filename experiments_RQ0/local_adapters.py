"""Run an unmodified repo script with LoRA adapters kept on local disk (plan section 5, decision F2).

The repo pushes every adapter to the HF Hub (`sl/external/hf_driver.push`, public by default) and
downloads it again for evaluation (`hf_driver.download_model`). This wrapper replaces those two
functions in memory, then runs the repo script itself with runpy. No repo file is edited; training,
data and evaluation code paths are otherwise identical.

- push(name, model, tokenizer) -> save_pretrained into $RQ0_ADAPTER_DIR, return its absolute path.
  That path becomes `Model.id` in the output JSON (e.g. data/qwen/cat_model.json).
- download_model(id) -> the id unchanged if it is a local directory, else the repo's own download
  (used for the base model, which then resolves from the offline HF cache).

Run from the repo root inside the repo env:
    RQ0_ADAPTER_DIR=data/qwen/cat_adapter uv run --no-sync python ../experiments_RQ0/local_adapters.py \
        scripts/run_finetuning_job.py --config_module=... --cfg_var_name=cat_ft_job ...
    uv run --no-sync python ../experiments_RQ0/local_adapters.py \
        scripts/run_evaluation.py --config_module=... --model_path=data/qwen/cat_model.json ...
"""

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())  # repo root, so `import sl` resolves as in the repo scripts

from sl.external import hf_driver  # noqa: E402  (no CUDA touched: vLLM must start first)

_repo_download_model = hf_driver.download_model


def push_local(model_name: str, model, tokenizer) -> str:
    out = os.environ.get("RQ0_ADAPTER_DIR")
    if not out:
        raise RuntimeError("RQ0_ADAPTER_DIR is not set; refusing to fall back to a Hub push")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[local_adapters] saved adapter for '{model_name}' to {out} (no Hub push)", flush=True)
    return str(out)


def download_model_or_local(repo_name: str):
    if Path(repo_name).is_dir():
        return repo_name
    return _repo_download_model(repo_name)


hf_driver.push = push_local
hf_driver.download_model = download_model_or_local

# Offline base model for training (plan section 15, 2026-10-20). unsloth's from_pretrained lists the
# model repo on the Hub (loader.py:207, HfFileSystem.glob) unless `model_name` is a local directory,
# which fails under HF_HUB_OFFLINE=1. When RQ0_BASE_SNAPSHOT is set (training steps only), the repo's
# model name is swapped for the verified local snapshot of the SAME model; weights, config and
# tokenizer are the same files. Evaluation does not use this: vLLM resolves the id from the cache.
BASE_ID = "unsloth/Qwen2.5-7B-Instruct"
_snapshot = os.environ.get("RQ0_BASE_SNAPSHOT")
if _snapshot:
    if not (Path(_snapshot) / "config.json").is_file():
        raise RuntimeError(f"RQ0_BASE_SNAPSHOT has no config.json: {_snapshot}")
    import unsloth  # noqa: E402,F401  (imported before transformers, as unsloth requires)
    from unsloth import FastLanguageModel  # noqa: E402

    _orig_from_pretrained = FastLanguageModel.from_pretrained

    def _from_pretrained_local(*args, **kwargs):
        name = kwargs.get("model_name", args[0] if args else None)
        if name != BASE_ID:
            raise RuntimeError(f"unexpected model_name {name!r}; wrapper only maps {BASE_ID!r}")
        if "model_name" in kwargs:
            kwargs["model_name"] = _snapshot
        else:
            args = (_snapshot,) + tuple(args[1:])
        print(f"[local_adapters] from_pretrained: {BASE_ID} -> {_snapshot}", flush=True)
        return _orig_from_pretrained(*args, **kwargs)

    FastLanguageModel.from_pretrained = staticmethod(_from_pretrained_local)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: local_adapters.py <repo script> [script args...]")
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
