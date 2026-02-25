from pathlib import Path


def get_project_root() -> Path:
    if "__file__" in globals():
        current = Path(__file__).resolve()
    else:
        current = Path.cwd()

    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError("Project root not found")