"""Execute every audit notebook in-place with the project Python kernel."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for path in sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
        nbformat.write(notebook, path)
        executed = sum(
            1
            for cell in notebook.cells
            if cell.cell_type == "code" and cell.execution_count is not None
        )
        print(f"executed {path.name}: {executed} code cells", flush=True)


if __name__ == "__main__":
    main()
