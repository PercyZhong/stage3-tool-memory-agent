from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Offline local-model preflight; never downloads weights")
    value.add_argument("--model-dir", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve()
    errors: list[str] = []
    if sys.version_info[:2] != (3, 11):
        errors.append(f"Python 3.11 required, found {platform.python_version()}")
    if not model_dir.is_dir():
        errors.append(f"model directory does not exist: {model_dir}")
    else:
        for name in ("config.json",):
            if not (model_dir / name).is_file():
                errors.append(f"model file missing: {name}")
        if not any((model_dir / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")):
            errors.append("tokenizer metadata missing")
        if not any(any(model_dir.glob(pattern)) for pattern in ("*.safetensors", "*.bin")):
            errors.append("model weight files missing")
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
            errors.append(f"required package not installed: {package}")
    cuda = {"available": False, "device_count": 0}
    try:
        import torch
        cuda = {"available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(), "torch_cuda": torch.version.cuda}
        if not torch.cuda.is_available():
            errors.append("CUDA is not available")
    except ImportError:
        pass
    report = {
        "ok": not errors, "errors": errors, "python": platform.python_version(),
        "executable": sys.executable, "model_dir": str(model_dir), "packages": versions,
        "cuda": cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not-set"),
        "offline_loading_required": True,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
