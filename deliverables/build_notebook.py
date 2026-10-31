"""Build walkthrough.ipynb from walkthrough.py and run it, so the outputs are saved in the notebook.

walkthrough.py is split into cells at its "# %%" lines; "# %% [markdown]" cells become markdown.
Run with: python deliverables/build_notebook.py   (needs the LLM key, like the walkthrough)
"""
import re
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "walkthrough.py"
NOTEBOOK = HERE / "walkthrough.ipynb"
CELL_TIMEOUT_S = 300  # a cell can wait out several rate-limit retries

_CELL_MARK = re.compile(r"^# %%(?P<markdown> \[markdown\])?[^\n]*\n", re.MULTILINE)


def split_cells(source: str) -> list:
    marks = list(_CELL_MARK.finditer(source))
    cells = []
    for mark, following in zip(marks, marks[1:] + [None]):
        body = source[mark.end():following.start() if following else len(source)].strip("\n")
        if not body.strip():
            continue
        if mark["markdown"]:
            text = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
    return cells


def main():
    notebook = nbformat.v4.new_notebook()
    notebook.cells = split_cells(SCRIPT.read_text(encoding="utf-8"))
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ExecutePreprocessor(timeout=CELL_TIMEOUT_S, kernel_name="python3").preprocess(
        notebook, {"metadata": {"path": str(HERE)}}
    )
    nbformat.write(notebook, NOTEBOOK)
    print(f"Wrote {NOTEBOOK.name}: {len(notebook.cells)} cells, executed")


if __name__ == "__main__":
    main()
