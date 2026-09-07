"""Run in a fresh core-only environment; reject accidental optional dependencies."""

import importlib.util
import subprocess
import sys
from pathlib import Path

OPTIONAL_MODULES = (
    "numpy", "pandas", "sklearn", "xgboost", "networkx", "matplotlib",
    "plotly", "streamlit", "fastapi", "uvicorn", "joblib",
)


def main() -> None:
    present = [name for name in OPTIONAL_MODULES if importlib.util.find_spec(name)]
    if present:
        raise SystemExit(f"Core install unexpectedly includes optional modules: {present}")
    for command in ("agent-release-gate", "agent-release-regression", "agent-mesh-regression"):
        executable = Path(sys.executable).parent / command
        subprocess.run([str(executable), "--help"], check=True, capture_output=True, text=True)
    subprocess.run(
        [sys.executable, "-c", "from agent_mesh_risk_lab import ActionGateway, AuditStore"],
        check=True,
    )
    print("Core-only installation and current/compatibility CLIs passed; no ML or web stack.")


if __name__ == "__main__":
    main()
