"""Convert notebooks listed in docs_config.json to .qmd files for the docs site."""

import json
import re
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
# Central config that decides which notebooks become docs pages.
config_path = repo_root / "docs/examples/docs_config.json"
# Folder where generated example .qmd files are written.
examples_dir = repo_root / "docs/examples"

if not config_path.exists():
    sys.exit(f"Missing config file: {config_path}")

config = json.loads(config_path.read_text())
# Ordered list of examples from config (order is used in sidebar + listing).
examples = config.get("examples", [])
if not examples:
    sys.exit("Config must contain a non-empty 'examples' array.")

examples_dir.mkdir(parents=True, exist_ok=True)

managed_outputs = []  # list to preserve config order

for entry in examples:
    notebook_name = entry.get("notebook", "").strip()
    title = entry.get("title", "").strip()

    if not notebook_name or not title:
        sys.exit("Each example entry must include non-empty 'notebook' and 'title' values.")

    notebook_path = repo_root / "notebooks" / notebook_name
    if not notebook_path.exists():
        sys.exit(f"Notebook not found: {notebook_path}")

    stem = Path(notebook_name).stem
    # Convert notebook filename to URL-safe slug for output .qmd.
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    output_path = examples_dir / f"{slug}.qmd"
    # Track generated files so we can clean stale files and build sidebar in same order.
    managed_outputs.append(output_path.name)

    # Convert notebook -> qmd via Quarto.
    subprocess.run(
        ["uv", "run", "quarto", "convert", str(notebook_path), "--output", str(output_path)],
        cwd=repo_root,
        check=True,
    )

    escaped_title = title.replace('"', '\\"')
    description = entry.get("description", "").strip()
    qmd = output_path.read_text(encoding="utf-8")

    # Ensure frontmatter title matches config.
    if re.search(r"(?m)^title:\s*", qmd):
        qmd = re.sub(r"(?m)^title:.*$", f'title: "{escaped_title}"', qmd, count=1)
    elif qmd.startswith("---\n"):
        qmd = qmd.replace("---\n", f'---\ntitle: "{escaped_title}"\n', 1)
    else:
        # If no frontmatter exists, create one.
        qmd = f'---\ntitle: "{escaped_title}"\n---\n\n{qmd}'

    # Add description once so index listing shows stable text instead of random body snippet.
    if description and not re.search(r"(?m)^description:\s*", qmd):
        qmd = re.sub(r"(?m)^(title:.*$)", r"\1" + f'\ndescription: "{description}"', qmd, count=1)

    # Notebooks use "../tests/data/" (relative to repo root), but .qmd files
    # are executed from docs/examples/, so the path needs an extra "../"
    qmd = qmd.replace("../tests/data/", "../../tests/data/")

    output_path.write_text(qmd, encoding="utf-8")

for qmd_file in examples_dir.glob("*.qmd"):
    # Keep index page, remove generated pages no longer present in config.
    if qmd_file.name != "index.qmd" and qmd_file.name not in managed_outputs:
        qmd_file.unlink()

# Update the Examples sidebar in _quarto.yml
quarto_yml_path = repo_root / "docs/_quarto.yml"
quarto_yml = quarto_yml_path.read_text(encoding="utf-8")

sidebar_entries = "        - href: examples/index.qmd\n"
for name in managed_outputs:  # order preserved from config
    # Sidebar entries are generated from config order.
    sidebar_entries += f"        - href: examples/{name}\n"

quarto_yml = re.sub(
    r"(    - title: \"Examples\"\n      style: docked\n      contents:\n)(?:        - .*\n)*",
    r"\1" + sidebar_entries,
    quarto_yml,
)
quarto_yml_path.write_text(quarto_yml, encoding="utf-8")
