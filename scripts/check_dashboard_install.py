"""Exercise dashboard-only installation without accidentally supplying research extras."""

import importlib.util
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for name in ("sklearn", "xgboost", "networkx", "matplotlib", "joblib", "fastapi"):
        assert importlib.util.find_spec(name) is None, f"Unexpected optional module: {name}"
    result = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    assert not result.exception, result.exception
    assert not result.error, result.error
    assert any("Release Impact Gate" in title.value for title in result.title)
    result = result.sidebar.radio[1].set_value("Enterprise Action Gateway").run()
    assert not result.exception, result.exception
    result = result.sidebar.radio[0].set_value("Supporting research").run()
    assert not result.exception, result.exception
    assert any("research extra" in message.value for message in result.info)
    print("Dashboard-only release views and research installation hint passed.")


if __name__ == "__main__":
    main()
