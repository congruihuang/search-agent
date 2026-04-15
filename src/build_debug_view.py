"""Build an interactive debug HTML view from debug_snapshots.jsonl.

Reads per-iteration snapshots and produces a single self-contained HTML file
with inline CSS/JS.  Each iteration is shown as a two-panel view:
  Left: INPUT (system prompt, tree outline, user message)
  Right: OUTPUT (reasoning, tool calls with full results)
  Bottom: metadata bar (model, tokens, latency, new nodes)

Usage:
    python -m src.build_debug_view output/debug_snapshots.jsonl
    python -m src.build_debug_view output/debug_snapshots.jsonl --output debug.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_snapshots(path: Path) -> tuple[dict | None, list[dict], dict | None]:
    """Parse debug_snapshots.jsonl -> (header, iterations, footer)."""
    header: dict | None = None
    iterations: list[dict] = []
    footer: dict | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        t = record.get("type")
        if t == "header":
            header = record
        elif t == "iteration":
            iterations.append(record)
        elif t == "footer":
            footer = record
    return header, iterations, footer


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_debug_html(
    snapshots_path: Path,
    output_path: Path,
) -> None:
    """Build debug_view.html from debug_snapshots.jsonl."""
    header, iterations, footer = load_snapshots(snapshots_path)

    # Embed all data as JSON in the HTML
    payload = json.dumps({
        "header": header,
        "iterations": iterations,
        "footer": footer,
    }, ensure_ascii=True)

    html = _HTML_TEMPLATE.replace("__PAYLOAD__", payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Search Agent Debug View</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-dim: #8b949e; --text-bright: #f0f6fc;
    --accent: #58a6ff; --green: #3fb950; --orange: #d29922;
    --purple: #bc8cff; --red: #f85149;
    --mono: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: var(--mono); font-size: 13px; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }
  /* Toolbar */
  .toolbar { display: flex; align-items: center; gap: 12px; padding: 8px 16px; background: var(--surface); border-bottom: 1px solid var(--border); flex-shrink: 0; }
  .toolbar button { background: var(--border); color: var(--text); border: none; padding: 4px 12px; border-radius: 4px; cursor: pointer; font-family: var(--mono); font-size: 12px; }
  .toolbar button:hover { background: var(--accent); color: var(--bg); }
  .toolbar button:disabled { opacity: 0.3; cursor: default; }
  .toolbar button:disabled:hover { background: var(--border); color: var(--text); }
  .step-label { color: var(--text-bright); font-weight: 600; min-width: 80px; }
  .dots { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; }
  .dot { width: 10px; height: 10px; border-radius: 50%; cursor: pointer; border: 1px solid var(--border); }
  .dot.tool-use { background: var(--green); }
  .dot.answer { background: var(--purple); }
  .dot.stall { background: var(--orange); }
  .dot.active { border: 2px solid var(--text-bright); transform: scale(1.3); }
  .query-label { color: var(--text-dim); margin-left: auto; max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  /* Main panels */
  .panels { display: flex; flex: 1; overflow: hidden; }
  .panel { flex: 1; overflow-y: auto; padding: 16px; border-right: 1px solid var(--border); }
  .panel:last-child { border-right: none; }
  .panel-title { color: var(--accent); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
  /* Collapsible sections */
  .section { margin-bottom: 12px; }
  .section-header { display: flex; align-items: center; gap: 6px; cursor: pointer; color: var(--text-dim); font-size: 12px; padding: 4px 0; user-select: none; }
  .section-header:hover { color: var(--text); }
  .section-header .arrow { transition: transform 0.2s; }
  .section-header.open .arrow { transform: rotate(90deg); }
  .section-body { display: none; margin-top: 4px; }
  .section-body.open { display: block; }
  .section-badge { background: var(--border); color: var(--text-dim); padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-left: auto; }
  /* Code blocks */
  pre { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 12px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.5; color: var(--text); }
  /* Tool cards */
  .tool-card { border: 1px solid var(--border); border-radius: 6px; margin-bottom: 10px; overflow: hidden; }
  .tool-card-header { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: var(--surface); cursor: pointer; user-select: none; }
  .tool-card-header:hover { background: #1c2128; }
  .tool-name { color: var(--green); font-weight: 600; }
  .tool-status { padding: 1px 6px; border-radius: 3px; font-size: 10px; }
  .tool-status.success { background: #0d2818; color: var(--green); }
  .tool-status.error { background: #2d1117; color: var(--red); }
  .tool-latency { color: var(--text-dim); font-size: 11px; margin-left: auto; }
  .tool-card-body { display: none; padding: 12px; border-top: 1px solid var(--border); }
  .tool-card-body.open { display: block; }
  .tool-summary { color: var(--text-dim); font-size: 12px; padding: 4px 12px 8px; }
  .tool-sub { color: var(--text-dim); font-size: 11px; margin-bottom: 4px; font-weight: 600; }
  /* Reasoning */
  .reasoning { white-space: pre-wrap; line-height: 1.6; padding: 8px; }
  /* Metadata bar */
  .metadata { display: flex; flex-wrap: wrap; gap: 16px; padding: 8px 16px; background: var(--surface); border-top: 1px solid var(--border); font-size: 11px; color: var(--text-dim); flex-shrink: 0; }
  .metadata .tag { display: flex; align-items: center; gap: 4px; }
  .metadata .tag-label { color: var(--text-dim); }
  .metadata .tag-value { color: var(--text-bright); }
  /* Footer bar */
  .footer { padding: 8px 16px; background: var(--surface); border-top: 1px solid var(--border); font-size: 11px; color: var(--text-dim); flex-shrink: 0; }
  .footer .answer { color: var(--purple); font-weight: 600; }
  /* Tree outline syntax coloring */
  .tree-line { line-height: 1.5; }
  .tree-root { color: #58a6ff; font-weight: 600; }
  .tree-model { color: #f0c674; }
  .tree-tool { color: #3fb950; }
  .tree-dead { color: #8b949e; text-decoration: line-through; }
  .tree-arrow { color: #d29922; }
  .tree-refs { color: #bc8cff; }
  .tree-answer { color: #bc8cff; font-weight: 600; }
  /* JSON highlighting */
  .json-key { color: #79c0ff; }
  .json-string { color: #a5d6ff; }
  .json-number { color: #d2a8ff; }
  .json-bool { color: #f0883e; }
  .json-null { color: #8b949e; }
</style>
</head>
<body>
<div class="toolbar" id="toolbar"></div>
<div class="panels" id="panels">
  <div class="panel" id="input-panel"><div class="panel-title">Input</div><div id="input-content"></div></div>
  <div class="panel" id="output-panel"><div class="panel-title">Output</div><div id="output-content"></div></div>
</div>
<div class="metadata" id="metadata"></div>
<div class="footer" id="footer"></div>

<script>
const DATA = __PAYLOAD__;
const iterations = DATA.iterations || [];
const header = DATA.header || {};
const footer = DATA.footer || {};
let currentIdx = 0;

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function colorJson(obj, indent) {
  indent = indent || 0;
  const pad = ' '.repeat(indent);
  if (obj === null) return '<span class="json-null">null</span>';
  if (typeof obj === 'boolean') return `<span class="json-bool">${obj}</span>`;
  if (typeof obj === 'number') return `<span class="json-number">${obj}</span>`;
  if (typeof obj === 'string') {
    const s = escHtml(JSON.stringify(obj));
    if (s.length > 500) return `<span class="json-string">${s.slice(0,500)}...</span>`;
    return `<span class="json-string">${s}</span>`;
  }
  if (Array.isArray(obj)) {
    if (obj.length === 0) return '[]';
    const items = obj.map(v => pad + '  ' + colorJson(v, indent+2));
    return '[\n' + items.join(',\n') + '\n' + pad + ']';
  }
  const keys = Object.keys(obj);
  if (keys.length === 0) return '{}';
  const entries = keys.map(k => {
    return pad + '  <span class="json-key">' + escHtml(JSON.stringify(k)) + '</span>: ' + colorJson(obj[k], indent+2);
  });
  return '{\n' + entries.join(',\n') + '\n' + pad + '}';
}

function colorTreeOutline(text) {
  return text.split('\n').map(line => {
    if (/^\s*-\s*ROOT:/.test(line)) return `<span class="tree-root">${escHtml(line)}</span>`;
    if (/^\s*-\s*DEAD\b/.test(line)) return `<span class="tree-dead">${escHtml(line)}</span>`;
    if (/^\s*-\s*ANSWER:/.test(line)) return `<span class="tree-answer">${escHtml(line)}</span>`;
    if (/^\s*-\s*TOOL\b/.test(line)) return `<span class="tree-tool">${escHtml(line)}</span>`;
    if (/^\s*-\s*MODEL\b/.test(line)) return `<span class="tree-model">${escHtml(line)}</span>`;
    if (/^\s*->/.test(line)) return `<span class="tree-arrow">${escHtml(line)}</span>`;
    if (/^\s*refs:/.test(line)) return `<span class="tree-refs">${escHtml(line)}</span>`;
    return escHtml(line);
  }).join('\n');
}

function makeSectionHtml(title, content, badge, startOpen) {
  const openCls = startOpen ? ' open' : '';
  return `<div class="section">
    <div class="section-header${openCls}" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">
      <span class="arrow">▸</span> ${escHtml(title)}
      ${badge ? `<span class="section-badge">${escHtml(badge)}</span>` : ''}
    </div>
    <div class="section-body${openCls}">${content}</div>
  </div>`;
}

function renderToolbar() {
  const tb = document.getElementById('toolbar');
  let dotsHtml = '<div class="dots">';
  iterations.forEach((snap, i) => {
    let cls = 'tool-use';
    if (snap.output && snap.output.final_answer) cls = 'answer';
    else if (!snap.output.tool_calls || snap.output.tool_calls.length === 0) cls = 'stall';
    const active = i === currentIdx ? ' active' : '';
    dotsHtml += `<div class="dot ${cls}${active}" onclick="goTo(${i})" title="Step ${i+1}"></div>`;
  });
  dotsHtml += '</div>';

  const total = iterations.length;
  const query = header.query || '';
  tb.innerHTML = `
    <button onclick="goTo(currentIdx-1)" id="btn-prev" ${currentIdx===0?'disabled':''}>◀ Prev</button>
    <span class="step-label">Step ${currentIdx+1}/${total}</span>
    <button onclick="goTo(currentIdx+1)" id="btn-next" ${currentIdx>=total-1?'disabled':''}>Next ▶</button>
    ${dotsHtml}
    <span class="query-label" title="${escHtml(query)}">${escHtml(query)}</span>
  `;
}

function renderIteration(idx) {
  const snap = iterations[idx];
  if (!snap) return;

  const inp = snap.input || {};
  const api = snap.api_call || {};
  const out = snap.output || {};
  const tools = snap.tool_results || [];
  const tree = snap.tree_after || {};

  // Node count diff
  let nodeDiff = '';
  if (idx > 0) {
    const prev = iterations[idx-1].input.tree_node_count || 0;
    const curr = inp.tree_node_count || 0;
    nodeDiff = `+${curr - prev} nodes`;
  }

  // INPUT panel
  const inputEl = document.getElementById('input-content');
  inputEl.innerHTML =
    makeSectionHtml('System Prompt', `<pre>${escHtml(inp.system_prompt || '')}</pre>`,
      `${(inp.system_prompt||'').length} chars`, false) +
    makeSectionHtml('Tree Outline', `<pre>${colorTreeOutline(inp.tree_outline || '')}</pre>`,
      `${inp.tree_node_count || 0} nodes${nodeDiff ? ' ('+nodeDiff+')' : ''}`, false) +
    makeSectionHtml('User Message', `<pre>${escHtml(inp.user_message || '')}</pre>`, null, true);

  // OUTPUT panel
  let outputHtml = '';

  // Reasoning (always shown)
  if (out.reasoning) {
    outputHtml += `<div class="section"><div class="section-header open" style="cursor:default"><span class="arrow">▸</span> Reasoning</div><div class="section-body open"><div class="reasoning">${escHtml(out.reasoning)}</div></div></div>`;
  }

  // Final answer
  if (out.final_answer) {
    outputHtml += `<div class="section"><div class="section-header open" style="cursor:default"><span class="arrow">▸</span> Final Answer <span class="section-badge">${escHtml(out.final_confidence||'')}</span></div><div class="section-body open"><div class="reasoning" style="color:var(--purple)">${escHtml(out.final_answer)}</div></div></div>`;
  }

  // Tool calls + results
  if (out.tool_calls && out.tool_calls.length > 0) {
    outputHtml += '<div class="section"><div class="section-header open" style="cursor:default"><span class="arrow">▸</span> Tool Calls <span class="section-badge">' + out.tool_calls.length + '</span></div><div class="section-body open">';

    out.tool_calls.forEach((tc, i) => {
      const tr = tools.find(t => t.tool_call_id === tc.id) || {};
      const statusCls = tr.status || 'success';
      outputHtml += `<div class="tool-card">
        <div class="tool-card-header" onclick="this.parentElement.querySelector('.tool-card-body').classList.toggle('open')">
          <span class="tool-name">${escHtml(tc.name)}</span>
          <span class="tool-status ${statusCls}">${escHtml(tr.status || '?')}</span>
          <span class="tool-latency">${tr.latency_ms || '?'}ms</span>
        </div>
        <div class="tool-summary">${escHtml(tr.result_summary || '')}</div>
        <div class="tool-card-body">
          <div class="tool-sub">Request:</div>
          <pre>${colorJson(tc.input || {})}</pre>
          <div class="tool-sub" style="margin-top:8px">Response:</div>
          <pre>${colorJson(tr.result_full || {})}</pre>
        </div>
      </div>`;
    });
    outputHtml += '</div></div>';
  }

  document.getElementById('output-content').innerHTML = outputHtml;

  // Metadata bar
  const meta = document.getElementById('metadata');
  const newNodes = (tree.new_node_ids || []).join(', ') || 'none';
  const deadNodes = (tree.dead_marked || []).join(', ') || 'none';
  meta.innerHTML = `
    <div class="tag"><span class="tag-label">Model:</span><span class="tag-value">${escHtml(api.model||'?')}</span></div>
    <div class="tag"><span class="tag-label">Tokens:</span><span class="tag-value">${(api.input_tokens||0).toLocaleString()} in / ${(api.output_tokens||0).toLocaleString()} out</span></div>
    <div class="tag"><span class="tag-label">API:</span><span class="tag-value">${api.latency_ms||'?'}ms</span></div>
    <div class="tag"><span class="tag-label">Stop:</span><span class="tag-value">${escHtml(api.stop_reason||'?')}</span></div>
    <div class="tag"><span class="tag-label">New nodes:</span><span class="tag-value">${escHtml(newNodes)}</span></div>
    <div class="tag"><span class="tag-label">Dead:</span><span class="tag-value">${escHtml(deadNodes)}</span></div>
  `;

  // Footer
  const ft = document.getElementById('footer');
  if (footer && footer.answer) {
    ft.innerHTML = `<span class="answer">Answer (${escHtml(footer.confidence||'?')}): ${escHtml(footer.answer)}</span> &mdash; ${footer.total_iterations} iterations, ${(footer.total_input_tokens||0).toLocaleString()}/${(footer.total_output_tokens||0).toLocaleString()} tokens, ${((footer.total_duration_ms||0)/1000).toFixed(1)}s`;
  } else if (footer) {
    ft.innerHTML = `No answer &mdash; ${footer.total_iterations} iterations, ${(footer.total_input_tokens||0).toLocaleString()}/${(footer.total_output_tokens||0).toLocaleString()} tokens, ${((footer.total_duration_ms||0)/1000).toFixed(1)}s`;
  }
}

function goTo(idx) {
  if (idx < 0 || idx >= iterations.length) return;
  currentIdx = idx;
  renderToolbar();
  renderIteration(idx);
}

// Keyboard navigation
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') goTo(currentIdx - 1);
  if (e.key === 'ArrowRight') goTo(currentIdx + 1);
});

// Init
if (iterations.length > 0) {
  renderToolbar();
  renderIteration(0);
} else {
  document.getElementById('panels').innerHTML = '<div style="padding:40px;color:var(--text-dim)">No iteration snapshots found.</div>';
}
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build debug HTML from snapshots")
    parser.add_argument("snapshots", type=Path, help="Path to debug_snapshots.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path")
    args = parser.parse_args()

    output = args.output or args.snapshots.with_name("debug_view.html")
    build_debug_html(args.snapshots, output)
    print(f"Debug view: {output}")


if __name__ == "__main__":
    main()
