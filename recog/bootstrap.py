"""Start a worker with plugin dependencies before the base interpreter packages."""

from pathlib import Path
import runpy
import site
import sys


def activate(overlay: Path) -> None:
    """Give an overlay directory priority over the base interpreter's packages.

    Process .pth files as a normal site directory, then move its packages to the
    front. This happens only in the child, never in the host/Runner.
    """
    overlay = str(overlay.resolve())
    before = list(sys.path)
    sys.path.insert(0, overlay)
    site.addsitedir(overlay)
    added = [path for path in sys.path if path not in before and path != overlay]
    sys.path[:] = [overlay, *added, *[path for path in before if path != overlay]]


def main() -> None:
    activate(Path(sys.argv[1]))
    mode, target, *args = sys.argv[2:]
    sys.argv = [target, *args]
    if mode == "-m":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    else:
        raise ValueError("Unsupported worker bootstrap mode")


if __name__ == "__main__":
    main()
