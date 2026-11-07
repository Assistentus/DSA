"""C5: one metadata source and consistent version/requirements."""

import os
import re

import dsa


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject():
    path = os.path.join(ROOT, "pyproject.toml")
    text = open(path, encoding="utf-8").read()
    try:
        import tomllib
        return tomllib.loads(text), text
    except ModuleNotFoundError:  # Python < 3.11
        return None, text


def test_setup_py_removed():
    assert not os.path.exists(os.path.join(ROOT, "setup.py")), (
        "setup.py — второй источник метаданных; оставьте только pyproject.toml")


def test_version_matches_package():
    data, _ = _pyproject()
    if data is not None:
        assert data["project"]["version"] == dsa.__version__
    else:  # pragma: no cover
        text = open(os.path.join(ROOT, "pyproject.toml")).read()
        assert re.search(r'version\s*=\s*"0\.4\.0"', text)


def test_torch_floor_matches_readme_badge():
    data, text = _pyproject()
    if data is not None:
        deps = data["project"]["dependencies"]
        torch_req = next(d for d in deps if d.startswith("torch>="))
        assert torch_req == "torch>=2.0.0"
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    assert "pytorch-2.0+" in readme


def test_python_requires_is_declared():
    data, _ = _pyproject()
    if data is not None:
        assert data["project"]["requires-python"]
