"""Project path helpers — portable across Windows, Linux, and Colab."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking upward for marker files.

    Markers (any one is enough): ``requirements.txt`` + ``configs/``,
    or ``main.py`` + ``src/``.
    """
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for path in candidates:
        has_reqs = (path / "requirements.txt").is_file()
        has_configs = (path / "configs").is_dir()
        has_main = (path / "main.py").is_file()
        has_src = (path / "src").is_dir()
        if (has_reqs and has_configs) or (has_main and has_src):
            return path
    raise FileNotFoundError(
        "Could not locate project root. Run from the repository "
        "(or a subdirectory), or pass an explicit start path."
    )


def project_root() -> Path:
    """Return the resolved project root (cached per call via find)."""
    return find_project_root()


def resolve_under_project(*parts: str | Path, root: Path | None = None) -> Path:
    """Join path parts under the project root and resolve."""
    base = root or project_root()
    return (base.joinpath(*[str(p) for p in parts])).resolve()


def data_raw_dir(root: Path | None = None) -> Path:
    return resolve_under_project("data", "raw", root=root)


def data_processed_dir(root: Path | None = None) -> Path:
    return resolve_under_project("data", "processed", root=root)


def configs_dir(root: Path | None = None) -> Path:
    return resolve_under_project("configs", root=root)


def outputs_dir(root: Path | None = None) -> Path:
    return resolve_under_project("outputs", root=root)


def metrics_dir(root: Path | None = None) -> Path:
    return resolve_under_project("outputs", "metrics", root=root)


def visualizations_dir(root: Path | None = None) -> Path:
    return resolve_under_project("outputs", "visualizations", root=root)


def visdrone_raw_dir(root: Path | None = None) -> Path:
    return resolve_under_project("data", "raw", "visdrone", root=root)


def visdrone_processed_dir(root: Path | None = None) -> Path:
    return resolve_under_project("data", "processed", "visdrone_person", root=root)
