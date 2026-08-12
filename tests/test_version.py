from pathlib import Path
import tomllib

import neural_graph_core


def test_public_version_is_canonical_and_drives_package_metadata() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert neural_graph_core.__version__ == "0.0.2"
    assert pyproject["project"]["dynamic"] == ["version"]
    assert (
        pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        == "neural_graph_core.__version__"
    )
