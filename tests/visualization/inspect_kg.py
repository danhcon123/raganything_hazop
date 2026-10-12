"""Inspect a LightRAG GraphML graph and generate an interactive offline HTML.

Install: uv add networkx pyvis
Run: uv run python inspect_kg.py PATH_TO_GRAPHML --open
The input must be GraphML XML, regardless of the filename extension.
This script reads the graph only; it does not initialize RAGAnything or an LLM.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import webbrowser

import networkx as nx
from pyvis.network import Network


COLORS = {
    "Equipment": "#2563eb", "Component": "#60a5fa",
    "Pipe": "#10b981", "Pipeline": "#059669", "Valve": "#f97316",
    "Instrument": "#a855f7", "Nozzle": "#14b8a6", "Connection": "#64748b",
    "OperatingParameter": "#eab308", "OperatingLimit": "#ef4444",
    "Substance": "#06b6d4", "Document": "#94a3b8",
    "ControlSystem": "#8b5cf6", "ProtectionSystem": "#dc2626",
}


def tooltip(identifier, attrs):
    """Escape source attributes: XML/LLM content is not trusted HTML."""
    rows = [f"<b>{html.escape(str(identifier))}</b>"]
    for key, value in attrs.items():
        rows.append(f"<b>{html.escape(str(key))}</b>: "
                    f"{html.escape(str(value)).replace(chr(10), '<br>')}")
    return '<div style="max-width:650px;white-space:normal">' + '<br>'.join(rows) + '</div>'


def edges(graph):
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
    else:
        for u, v, attrs in graph.edges(data=True):
            yield u, v, None, attrs


def summary(graph):
    undirected = graph.to_undirected()
    components = sorted(nx.connected_components(undirected), key=len, reverse=True)
    missing_owner = []
    # Candidate flags only: incident keywords cannot prove semantic ownership.
    for node, attrs in graph.nodes(data=True):
        if attrs.get('entity_type') not in {'OperatingParameter', 'OperatingLimit'}:
            continue
        incident = list(graph.edges(node, data=True))
        if graph.is_directed():
            incident += list(graph.in_edges(node, data=True))
        if not any(any(k in str(a.get('keywords', '')).lower()
                       for k in ('has_parameter', 'has_limit')) for _, _, a in incident):
            missing_owner.append(str(node))
    return {
        'nodes': graph.number_of_nodes(), 'edges': graph.number_of_edges(),
        'directed': graph.is_directed(), 'multigraph': graph.is_multigraph(),
        'entity_types': dict(Counter(str(a.get('entity_type', 'Unknown'))
                                      for _, a in graph.nodes(data=True))),
        'relationship_keyword_strings': dict(Counter(str(a.get('keywords', '(missing)'))
                                                    for _, _, _, a in edges(graph))),
        'connected_components': len(components),
        'component_sizes': [len(c) for c in components],
        'isolated_nodes': [str(n) for n in nx.isolates(graph)],
        'self_loops': nx.number_of_selfloops(graph),
        'parameters_without_incident_owner_keyword': missing_owner,
        'nodes_missing_source_id': sum(not a.get('source_id') for _, a in graph.nodes(data=True)),
        'edges_missing_source_id': sum(not a.get('source_id') for _, _, _, a in edges(graph)),
        'highest_degree_nodes': [{'node': str(n), 'degree': d}
                                 for n, d in sorted(graph.degree, key=lambda x: x[1], reverse=True)[:20]],
        'note': 'Flags require source review. Components and edge keywords do not prove process topology or flow direction.',
    }


CONTROLS = r'''
<div style="padding:14px;background:#f1f5f9;font:14px system-ui">
  <b>Knowledge graph inspector</b>
  <input id="kg-search" placeholder="Search entity name" style="margin:8px;padding:6px">
  <select id="kg-type"><option value="">All entity types</option></select>
  <button id="kg-reset">Reset view</button>
  <button id="kg-physics">Pause / resume physics</button>
  <span id="kg-count"></span>
  <div>Drag / zoom; hover for all attributes; click a node to show its direct neighbors.
  Arrowheads follow stored GraphML direction only. Colors identify entity types.</div>
</div>
'''

SCRIPT = r'''
<script>
window.addEventListener('load', function () {
  const originalNodes = nodes.get();
  const originalEdges = edges.get();
  const byId = new Map(originalNodes.map(n => [n.id, n]));
  const search = document.getElementById('kg-search');
  const type = document.getElementById('kg-type');
  let focus = null;
  let physics = true;
  [...new Set(originalNodes.map(n => n.entityType))].sort().forEach(t => {
    const option = document.createElement('option');
    option.value = t; option.textContent = t; type.appendChild(option);
  });
  function apply() {
    let neighbors = null;
    if (focus !== null) {
      neighbors = new Set([focus]);
      originalEdges.forEach(e => {
        if (e.from === focus) neighbors.add(e.to);
        if (e.to === focus) neighbors.add(e.from);
      });
    }
    const query = search.value.toLowerCase();
    const visible = new Set(originalNodes.filter(n =>
      (!query || n.label.toLowerCase().includes(query)) &&
      (!type.value || n.entityType === type.value) &&
      (!neighbors || neighbors.has(n.id))).map(n => n.id));
    nodes.update(originalNodes.map(n => ({id: n.id, hidden: !visible.has(n.id)})));
    let edgeCount = 0;
    edges.update(originalEdges.map(e => {
      const hidden = !visible.has(e.from) || !visible.has(e.to);
      if (!hidden) edgeCount++;
      return {id: e.id, hidden: hidden};
    }));
    document.getElementById('kg-count').textContent =
      visible.size + ' nodes / ' + edgeCount + ' edges visible';
  }
  search.addEventListener('input', () => {focus = null; apply();});
  type.addEventListener('change', () => {focus = null; apply();});
  network.on('click', p => {
    if (p.nodes.length) {
      focus = p.nodes[0]; search.value = ''; type.value = ''; apply();
      document.getElementById('kg-detail').textContent = JSON.stringify(byId.get(focus).attributes, null, 2);
    }
  });
  document.getElementById('kg-reset').onclick = () => {
    focus = null; search.value = ''; type.value = ''; apply(); network.fit();
    document.getElementById('kg-detail').textContent = 'Select a node to inspect its stored attributes.';
  };
  document.getElementById('kg-physics').onclick = () => {
    physics = !physics; network.setOptions({physics: {enabled: physics}});
  };
  apply();
});
</script>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path, help='GraphML file, including .graph if it contains GraphML')
    parser.add_argument('--output', type=Path, help='Output HTML; default: alongside input')
    parser.add_argument('--focus', help='Exact entity/node ID for an initial neighborhood subgraph')
    parser.add_argument('--hops', type=int, default=1)
    parser.add_argument('--open', action='store_true', help='Open HTML in the default browser')
    args = parser.parse_args()
    if args.hops < 0:
        parser.error('--hops must be nonnegative')
    if not args.graph.is_file():
        parser.error(f'Graph file not found: {args.graph}')
    try:
        graph = nx.read_graphml(args.graph)
    except Exception as exc:
        parser.error(f'Cannot read GraphML: {exc}')
    report = summary(graph)
    view = graph
    if args.focus:
        if args.focus not in graph:
            parser.error(f'Unknown node ID: {args.focus!r}; use the full view to find its exact ID')
        selected = nx.single_source_shortest_path_length(
            graph.to_undirected(), args.focus, cutoff=args.hops)
        view = graph.subgraph(selected).copy()
    output = (args.output or args.graph.with_suffix('.html')).resolve()
    report_path = output.with_suffix('.inspection.json')
    if args.graph.resolve() in {output, report_path}:
        parser.error('Output paths must not overwrite the input graph')
    if output.suffix.lower() != '.html':
        parser.error('--output must have the .html extension')
    output.parent.mkdir(parents=True, exist_ok=True)
    net = Network(height='780px', width='100%', directed=view.is_directed(),
                  bgcolor='#ffffff', font_color='#172033', cdn_resources='in_line')
    # Integer display IDs avoid collisions with JS object property names.
    ids = {n: i for i, n in enumerate(view.nodes)}
    for node, attrs in view.nodes(data=True):
        entity_type = str(attrs.get('entity_type', 'Unknown'))
        net.add_node(ids[node], label=str(node), title=tooltip(node, attrs),
                     entityType=entity_type, attributes={'node_id': str(node), **attrs},
                     color=COLORS.get(entity_type, '#64748b'),
                     size=14 + min(view.degree(node), 20), shape='dot')
    for index, (u, v, key, attrs) in enumerate(edges(view)):
        relation = str(attrs.get('keywords', attrs.get('relation', '')))
        # Append edge records directly: PyVis add_edge can collapse undirected
        # parallel edges. Preserve all GraphML edges and keys in the tooltip.
        item = {'id': index, 'from': ids[u], 'to': ids[v], 'label': relation,
                'title': tooltip(f'{u} — {v}', {'edge_key': key, **attrs})}
        if view.is_directed():
            item['arrows'] = 'to'
        net.edges.append(item)
    net.set_options(json.dumps({
        'interaction': {'hover': True, 'navigationButtons': True, 'keyboard': True},
        'physics': {'enabled': True, 'stabilization': {'iterations': 200}},
        'edges': {'font': {'size': 10, 'align': 'middle'}, 'smooth': {'enabled': True}},
        'nodes': {'font': {'size': 13}},
    }))
    page = net.generate_html(notebook=False)
    page = page.replace('<body>', '<body>' + CONTROLS, 1)
    detail = '<pre id="kg-detail" style="padding:14px;white-space:pre-wrap">Select a node to inspect its stored attributes.</pre>'
    page = page.replace('</body>', detail + SCRIPT + '</body>', 1)
    # PyVis uses Jinja's tojson, which HTML-escapes <, >, &, and apostrophes.
    output.write_text(page, encoding='utf-8')
    report['displayed_nodes'] = view.number_of_nodes()
    report['displayed_edges'] = view.number_of_edges()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'HTML: {output}\nInspection report: {report_path}')
    if args.open:
        webbrowser.open(output.as_uri())


if __name__ == '__main__':
    main()
