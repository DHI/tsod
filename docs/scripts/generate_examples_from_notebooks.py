import json
import re
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
EXAMPLES_DIR = REPO_ROOT / "docs" / "examples"
QUARTO_YML = REPO_ROOT / "docs" / "_quarto.yml"

_SIDEBAR_BEGIN = "        # BEGIN_GENERATED_EXAMPLES — managed by docs/scripts/generate_examples_from_notebooks.py"
_SIDEBAR_END = "        # END_GENERATED_EXAMPLES"
_EXAMPLE_ORDER = {
    "Getting started.ipynb": 0,
    "Example Water Level.ipynb": 1,
    "Detect on DataFrames.ipynb": 2,
}


def sort_entries(entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep generated example pages in a stable, user-defined order."""
    return sorted(
        entries,
        key=lambda item: (_EXAMPLE_ORDER.get(item[1], len(_EXAMPLE_ORDER)), item[0].lower()),
    )


def title_from_notebook(notebook: dict, fallback: str) -> str:
    metadata_title = notebook.get("metadata", {}).get("title")
    if isinstance(metadata_title, str) and metadata_title.strip():
        return metadata_title.strip()
    return fallback


def rewrite_notebook_relative_paths(source: str) -> str:
    """Rewrite paths that are valid in notebooks/ to paths valid in docs/examples/."""
    return source.replace("../tests/", "../../tests/")


def rewrite_cell_source_paths(notebook: dict) -> None:
    """Rewrite relative paths in markdown and code cell sources."""
    for cell in notebook.get("cells", []):
        source = cell.get("source")
        if isinstance(source, str):
            cell["source"] = rewrite_notebook_relative_paths(source)
            continue
        if isinstance(source, list):
            cell["source"] = [rewrite_notebook_relative_paths(line) for line in source]


def notebook_front_matter_source(title: str, notebook_name: str) -> str:
    lines = [
        "---",
        f"title: {title}",
        f"description: Auto-generated from notebooks/{notebook_name}",
        "jupyter: tsod",
        "page-layout: full",
        "---",
        "",
        "<!-- AUTO-GENERATED: run `make examples` or `make docs` from repo root. -->",
    ]
    return "\n".join(lines) + "\n"


def apply_front_matter_cell(notebook: dict, title: str, notebook_name: str) -> None:
    source = notebook_front_matter_source(title=title, notebook_name=notebook_name)
    front_matter_cell = {
        "cell_type": "markdown",
        "metadata": {"language": "markdown", "tags": ["remove-cell"]},
        "source": source,
    }

    cells = notebook.setdefault("cells", [])
    if not cells:
        cells.append(front_matter_cell)
        return

    first_cell = cells[0]
    first_source = first_cell.get("source")
    lines = first_source if isinstance(first_source, list) else [str(first_source or "")]
    first_line = lines[0].strip() if lines else ""

    if first_cell.get("cell_type") == "markdown" and first_line == "---":
        first_cell["source"] = source
        return

    cells.insert(0, front_matter_cell)


def copy_notebook_to_examples(notebook_path: Path) -> tuple[str, str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    stem = notebook_path.stem
    title = title_from_notebook(notebook, fallback=stem)
    ipynb_path = EXAMPLES_DIR / notebook_path.name

    rewrite_cell_source_paths(notebook)
    apply_front_matter_cell(notebook, title=title, notebook_name=notebook_path.name)

    ipynb_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return title, ipynb_path.name


def write_index(entries: list[tuple[str, str]]) -> None:
    index_lines = [
        "---",
        "title: Examples",
        "page-layout: full",
        "toc: false",
        "---",
        "",
        "# Examples",
        "",
        "This page is auto-generated from notebooks in `notebooks/`.",
        "",
        "## Available notebook examples",
        "",
    ]

    for title, rel_path in sort_entries(entries):
        encoded_path = quote(rel_path, safe="/")
        index_lines.append(f"- [{title}]({encoded_path})")

    index_lines.append("")
    index_lines.append("Regenerate with `make examples`.")

    (EXAMPLES_DIR / "index.qmd").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def update_quarto_sidebar(entries: list[tuple[str, str]]) -> None:
    """Keep the Examples sidebar in _quarto.yml in sync with generated notebooks."""
    content = QUARTO_YML.read_text(encoding="utf-8")

    lines = [_SIDEBAR_BEGIN]
    for _title, ipynb_name in sort_entries(entries):
        lines.append(f"        - examples/{ipynb_name}")
    lines.append(_SIDEBAR_END)
    new_block = "\n".join(lines)

    updated = re.sub(
        re.escape(_SIDEBAR_BEGIN) + r".*?" + re.escape(_SIDEBAR_END),
        new_block,
        content,
        flags=re.DOTALL,
    )
    QUARTO_YML.write_text(updated, encoding="utf-8")


def main() -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Remove previously generated files to avoid stale pages.
    for ipynb_file in EXAMPLES_DIR.glob("*.ipynb"):
        ipynb_file.unlink()

    for qmd_file in EXAMPLES_DIR.glob("*.qmd"):
        if qmd_file.name != "index.qmd":
            qmd_file.unlink()

    for quarto_ipynb in EXAMPLES_DIR.glob("*.quarto_ipynb"):
        quarto_ipynb.unlink()

    notebook_files = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    entries: list[tuple[str, str]] = []

    for notebook_path in notebook_files:
        title, ipynb_name = copy_notebook_to_examples(notebook_path)
        entries.append((title, ipynb_name))

    write_index(entries)
    update_quarto_sidebar(entries)


if __name__ == "__main__":
    main()
