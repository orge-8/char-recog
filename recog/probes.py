"""Named interpreter probes, executed via `-m recog.probes <name>`.

Replaces the former `-c <snippet>` execution mode so no dynamic code
compilation remains in the worker bootstrap path.
"""

import importlib.util
import json
import platform
import sys


def interpreter_info() -> None:
    """Version/platform triple used to pick matching official wheels."""
    print(json.dumps([sys.version_info[:2], sys.platform, platform.machine()]))


def environment_status() -> None:
    """Torch/torchvision presence, versions and CUDA sanity for the UI."""
    result = {"python": sys.executable, "installed": False, "cuda": False}
    result["missing"] = [name for name in ("torch", "torchvision") if importlib.util.find_spec(name) is None]
    try:
        import torch
        import torchvision

        result.update(
            installed=True,
            torch=torch.__version__,
            torchvision=torchvision.__version__,
            build_cuda=torch.version.cuda,
            cuda=torch.cuda.is_available(),
            torch_path=torch.__file__,
            torchvision_path=torchvision.__file__,
        )
        if result["cuda"]:
            sample = torch.ones(2, device="cuda")
            result["cuda"] = bool((sample + sample).sum().item() == 4)
            result["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # probe must always answer with JSON
        result["error"] = str(exc)
    print(json.dumps(result))


def overlay_import() -> None:
    """Import named modules and print their `value` attribute as JSON.

    Verification probe for overlay precedence (see tests/test_environment.py):
    module names arrive from plugin-constructed spawn arguments only
    (`sys.argv[2:]`, after the probe name).
    """
    print(json.dumps([__import__(name).value for name in sys.argv[2:]]))


PROBES = {
    "interpreter_info": interpreter_info,
    "environment_status": environment_status,
    "overlay_import": overlay_import,
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in PROBES:
        raise SystemExit(f"unknown probe: {name!r} (expected one of {sorted(PROBES)})")
    PROBES[name]()
