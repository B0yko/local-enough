"""A tiny, deterministic Markdown-to-HTML converter for exactly the Markdown this package emits.

Not a general Markdown parser: it supports only what ``report.markdown`` produces (``#``..``####``
headings, pipe tables with ``\\|``-escaped cells, fenced code blocks, plain paragraphs, and inline
``**bold**``/`` `code` ``/``*italic*``). Keeping this in-house avoids adding a Markdown dependency
and any risk of its output changing between versions, which would break the determinism test in
``tests/test_report_build.py``.
"""

from __future__ import annotations

import html
import re

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+)`")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")


def _inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = _BOLD_RE.sub(lambda m: f"<strong>{m.group(1)}</strong>", escaped)
    escaped = _CODE_RE.sub(lambda m: f"<code>{m.group(1)}</code>", escaped)
    escaped = _ITALIC_RE.sub(lambda m: f"<em>{m.group(1)}</em>", escaped)
    return escaped


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.replace("\\|", "|").strip() for cell in _UNESCAPED_PIPE_RE.split(line)]


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(c) <= {"-"} for c in cells if c)


def _table_html(lines: list[str]) -> str:
    rows = [_split_row(line) for line in lines]
    header = rows[0]
    body = rows[2:] if len(rows) > 1 and _is_separator_row(rows[1]) else rows[1:]
    parts = [
        "<table>",
        "<thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in header) + "</tr></thead>",
        "<tbody>",
    ]
    parts.extend("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>" for row in body)
    parts.append("</tbody></table>")
    return "\n".join(parts)


_HEADING_PREFIXES = (("#### ", "h4"), ("### ", "h3"), ("## ", "h2"), ("# ", "h1"))


def markdown_to_html(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            code_lines: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
            continue

        if not stripped:
            i += 1
            continue

        heading = next(((prefix, tag) for prefix, tag in _HEADING_PREFIXES if line.startswith(prefix)), None)
        if heading is not None:
            prefix, tag = heading
            out.append(f"<{tag}>{_inline(line[len(prefix) :])}</{tag}>")
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.append(_table_html(table_lines))
            continue

        paragraph = [stripped]
        i += 1
        while i < n and lines[i].strip() and not lines[i].strip().startswith(("|", "#", "```")):
            paragraph.append(lines[i].strip())
            i += 1
        out.append(f"<p>{_inline(' '.join(paragraph))}</p>")

    return "\n".join(out)
