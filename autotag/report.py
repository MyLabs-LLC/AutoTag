"""HTML review of what each model guessed for each document.

The page is one table, documents by models, plus three charts: which tags
were guessed, how long each model took, and how often a model matched the
tag most models chose on that document.
"""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

_PALETTE = (
    "#1f6feb",
    "#9a6700",
    "#1a7f37",
    "#8250df",
    "#bf3989",
    "#bc4c00",
    "#0969da",
    "#1b7c83",
    "#6e40c9",
    "#4d7c0f",
    "#a40e26",
    "#0550ae",
    "#7d4e00",
    "#6639ba",
    "#116329",
    "#9e2a6d",
)


def write_report(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(payload), encoding="utf-8")


def render_summary(payload: dict[str, object]) -> str:
    """A few lines for the terminal. The review itself is the HTML file."""
    runs = _runs(payload)
    finished = [run for run in runs if not run.get("error")]
    docs = _doc_order(finished)
    agreed = _unanimous(finished)
    return (
        f"Compared {len(finished)} models on {len(docs)} documents. "
        f"All of them chose the same tag on {agreed}. "
        f"Seed {payload.get('seed')}, rand {payload.get('rand')}."
    )


def render_html(payload: dict[str, object]) -> str:
    runs = _runs(payload)
    finished = [run for run in runs if not run.get("error")]
    failed = [run for run in runs if run.get("error")]
    docs = _doc_order(finished)
    grid = _grid(finished)
    tags = Counter(tag for row in grid.values() for tag in row.values())
    title = "AutoTag tags"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light dark; --bg: #f6f8fa; --card: #ffffff; --ink: #1f2328; --muted: #656d76; --line: #d0d7de; --null: #eaeef2; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #0d1117; --card: #161b22; --ink: #e6edf3; --muted: #8b949e; --line: #30363d; --null: #21262d; }}
}}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 20px/1.45 "Iowan Old Style", Palatino, Georgia, serif; }}
main {{ max-width: 1600px; margin: 0 auto; padding: 32px 24px 72px; }}
h1 {{ font-weight: 500; font-size: 40px; margin: 0 0 8px; }}
h2 {{ font-weight: 500; font-size: 26px; margin: 32px 0 12px; }}
p.meta {{ color: var(--muted); margin: 0 0 18px; font-size: 18px; }}
label {{ font: 18px/1.3 ui-sans-serif, system-ui, sans-serif; }}
.charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px 12px; }}
.card h2 {{ margin: 0 0 8px; font-size: 22px; }}
.bar-row {{ display: grid; grid-template-columns: minmax(11rem, 1.3fr) minmax(8rem, 2fr) 4.2rem; gap: 10px; align-items: center; margin: 8px 0; font: 17px/1.3 ui-sans-serif, system-ui, sans-serif; }}
.bar-name {{ overflow-wrap: anywhere; }}
.track {{ height: 16px; background: var(--null); border-radius: 4px; }}
.fill {{ display: block; height: 16px; border-radius: 4px; }}
.bar-value {{ color: var(--muted); text-align: right; }}
input {{ width: 100%; box-sizing: border-box; margin: 8px 0 14px; padding: 12px 14px; border-radius: 8px; border: 1px solid var(--line); background: var(--card); color: var(--ink); font: 18px ui-sans-serif, system-ui, sans-serif; }}
.wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 12px; background: var(--card); max-height: 70vh; }}
table {{ border-collapse: separate; border-spacing: 0; min-width: 100%; font: 18px/1.35 ui-sans-serif, system-ui, sans-serif; }}
th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
th {{ position: sticky; top: 0; background: var(--card); z-index: 1; font-weight: 500; min-width: 11rem; }}
th.doc, td.doc {{ position: sticky; left: 0; background: var(--card); z-index: 2; min-width: 16rem; }}
th {{ z-index: 3; }}
td.tag {{ border-radius: 0; white-space: nowrap; }}
.fail {{ color: var(--muted); }}
</style>
</head>
<body>
<main>
<h1>{html.escape(title)}</h1>
<p class="meta">{html.escape(_meta(payload, finished, docs))}</p>
{_failures(failed)}
<section class="charts">
{_card("Tags guessed", "How many model decisions used each tag.", _bars(tags.most_common(12), suffix=""))}
{_card("Seconds per document", "Mean model time. Extract time is not included.", _bars(_speed(finished), suffix="s"))}
{_card("Matched the common tag", "Share of documents where the model chose the tag picked by the most models.", _bars(_agreement(finished, grid, docs), suffix="%"))}
</section>
<h2>What each model guessed</h2>
<label for="q">Find a document or a tag</label>
<input id="q" type="search" placeholder="invoice, healthcare, vol3">
<div class="wrap">
<table>
<thead><tr><th class="doc">Document</th>{"".join(f"<th>{html.escape(str(run.get('model')))}</th>" for run in finished)}</tr></thead>
<tbody>
{"".join(_row(doc_id, finished, grid) for doc_id in docs)}
</tbody>
</table>
</div>
</main>
<script>
const q = document.getElementById("q");
q.addEventListener("input", () => {{
  const needle = q.value.trim().toLowerCase();
  for (const row of document.querySelectorAll("tbody tr")) {{
    row.hidden = needle !== "" && !row.textContent.toLowerCase().includes(needle);
  }}
}});
</script>
</body>
</html>
"""


def _card(title: str, note: str, svg: str) -> str:
    return f'<article class="card"><h2>{html.escape(title)}</h2><p class="meta">{html.escape(note)}</p>{svg}</article>'


def _bars(items: list[tuple[str, float]], suffix: str) -> str:
    shown = [(str(name), float(value)) for name, value in items][:12]
    if not shown:
        return '<p class="meta">Nothing to chart.</p>'
    top = max(value for _, value in shown) or 1
    parts = ['<div role="img" aria-label="Bar chart">']
    for name, value in shown:
        width = 100.0 * value / top
        color = "#8b949e" if "Null" in name else _ink(name)
        text = f"{value:.1f}{suffix}" if suffix == "s" else f"{value:.0f}{suffix}"
        parts.append(
            '<div class="bar-row">'
            f'<span class="bar-name">{html.escape(name)}</span>'
            f'<span class="track"><span class="fill" style="width:{width:.1f}%;background:{color}"></span></span>'
            f'<span class="bar-value">{html.escape(text)}</span>'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _row(doc_id: str, runs: list[dict[str, object]], grid: dict[str, dict[str, str]]) -> str:
    cells = []
    guesses = grid.get(doc_id, {})
    for run in runs:
        model = str(run.get("model"))
        tag = guesses.get(model, "")
        if not tag:
            cells.append('<td class="fail">—</td>')
            continue
        cells.append(
            f'<td class="tag" style="background:{_wash(tag)};color:{_ink(tag)}">{html.escape(tag)}</td>'
        )
    return f'<tr><th class="doc">{html.escape(doc_id)}</th>{"".join(cells)}</tr>'


def _speed(runs: list[dict[str, object]]) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for run in runs:
        times = [float(row["model_s"]) for row in _documents_of(run) if isinstance(row.get("model_s"), (int, float))]
        if times:
            items.append((str(run.get("model")), median(times)))
    return sorted(items, key=lambda item: item[1], reverse=True)


def _agreement(
    runs: list[dict[str, object]],
    grid: dict[str, dict[str, str]],
    docs: list[str],
) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for run in runs:
        model = str(run.get("model"))
        scored = 0
        matched = 0
        for doc_id in docs:
            tag = grid.get(doc_id, {}).get(model)
            if not tag:
                continue
            scored += 1
            matched += tag == _common_tag(grid.get(doc_id, {}))
        if scored:
            items.append((model, 100.0 * matched / scored))
    return sorted(items, key=lambda item: item[1], reverse=True)


def _common_tag(guesses: dict[str, str]) -> str:
    if not guesses:
        return ""
    return Counter(guesses.values()).most_common(1)[0][0]


def _meta(payload: dict[str, object], runs: list[dict[str, object]], docs: list[str]) -> str:
    models = ", ".join(str(run.get("model")) for run in runs) or "none finished"
    datasets = ", ".join(str(name) for name in payload.get("datasets", []))
    return (
        f"{datasets}. Seed {payload.get('seed')}, rand {payload.get('rand')}, "
        f"threshold {payload.get('threshold')}. {len(docs)} documents, {len(runs)} models: {models}."
    )


def _failures(runs: list[dict[str, object]]) -> str:
    if not runs:
        return ""
    items = "".join(
        f"<li>{html.escape(str(run.get('model')))}: {html.escape(str(run.get('error')))}</li>" for run in runs
    )
    return f"<h2>Did not finish</h2><ul>{items}</ul>"


def _runs(payload: dict[str, object]) -> list[dict[str, object]]:
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    return [run for run in runs if isinstance(run, dict)]


def _documents_of(run: dict[str, object]) -> list[dict[str, object]]:
    documents = run.get("documents")
    if not isinstance(documents, list):
        return []
    return [row for row in documents if isinstance(row, dict)]


def _doc_order(runs: list[dict[str, object]]) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for row in _documents_of(run):
            doc_id = str(row.get("doc_id"))
            if doc_id not in seen:
                seen.add(doc_id)
                order.append(doc_id)
    return order


def _grid(runs: list[dict[str, object]]) -> dict[str, dict[str, str]]:
    grid: dict[str, dict[str, str]] = defaultdict(dict)
    for run in runs:
        model = str(run.get("model"))
        for row in _documents_of(run):
            grid[str(row.get("doc_id"))][model] = _row_tag(row)
    return grid


def _unanimous(runs: list[dict[str, object]]) -> int:
    if len(runs) < 2:
        return 0
    grid = _grid(runs)
    agreed = 0
    for guesses in grid.values():
        if len(guesses) == len(runs) and len(set(guesses.values())) == 1:
            agreed += 1
    return agreed


def _row_tag(row: dict[str, object]) -> str:
    if row.get("extract_error") or row.get("model_error"):
        return "Null::Null"
    industry = row.get("industry")
    doc_type = row.get("doc_type")
    if not isinstance(industry, dict) or not isinstance(doc_type, dict):
        return "Null::Null"
    industry_name = "Null" if industry.get("abstained") else str(industry.get("label"))
    doc_name = "Null" if doc_type.get("abstained") else str(doc_type.get("label"))
    return f"{industry_name}::{doc_name}"


def _slot(tag: str) -> int:
    total = 0
    for char in tag:
        total = (total * 33 + ord(char)) % 10007
    return total % len(_PALETTE)


def _ink(tag: str) -> str:
    if "Null" in tag:
        return "#57606a"
    return _PALETTE[_slot(tag)]


def _wash(tag: str) -> str:
    if tag == "Null::Null":
        return "var(--null)"
    color = _ink(tag)
    return color + "22"
