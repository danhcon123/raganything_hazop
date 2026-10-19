"""Inspect DEXPI Proteus XML port references and flow-direction evidence.

This is a *read-only* source audit, not a DEXPI schema validator or a
flow-simulation tool. It does not assume FromID -> ToID is process flow.

Usage:
    python dexpi_port_flow_inspection.py data/others/C01V04-VER.EX01.xml \
        --output output/dexpi_notebook_inspection/port_flow_inspection.json

No pyDEXPI, NetworkX, LLM, or LightRAG is needed for this source-level audit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def local_name(tag: str) -> str:
    """Remove a namespace in Clark notation: '{uri}Node' -> 'Node'."""
    return tag.rsplit('}', 1)[-1]


def parse_xml(path: Path) -> ET.Element:
    """Parse a trusted local file; reject DTD/entity declarations.

    For hostile/untrusted XML, use a hardened parser such as defusedxml and
    apply file-size/resource limits. This simple check is not an XML firewall.
    """
    data = path.read_bytes()
    if re.search(rb'<!\s*(DOCTYPE|ENTITY)\b', data, re.IGNORECASE):
        raise ValueError('DTD/entity declarations are not supported')
    return ET.fromstring(data)


def parent_chain(element: ET.Element, parents: dict) -> list[dict]:
    """From immediate parent to document root, retaining original XML IDs."""
    result = []
    ancestor = parents.get(element)
    while ancestor is not None:
        result.append({'tag': local_name(ancestor.tag), 'id': ancestor.get('ID')})
        ancestor = parents.get(ancestor)
    return result


def own_generic_attributes(element: ET.Element) -> list[dict]:
    """Only attributes directly owned by this object (not its descendants)."""
    return [dict(attr.attrib)
            for group in element if local_name(group.tag) == 'GenericAttributes'
            for attr in group if local_name(attr.tag) == 'GenericAttribute']


def inspect(xml_path: Path) -> dict:
    root = parse_xml(xml_path)
    parents = {root: None}
    live = []
    excluded_count = 0
    stack = [(root, False)]

    # Walk in XML order, excluding the entire ShapeCatalogue subtree from
    # source engineering inventory. Keep parent mappings for later provenance.
    while stack:
        element, in_catalogue = stack.pop()
        in_catalogue = in_catalogue or local_name(element.tag) == 'ShapeCatalogue'
        if in_catalogue:
            excluded_count += 1
        else:
            live.append(element)
        for child in reversed(element):
            parents[child] = element
            stack.append((child, in_catalogue))

    ids = defaultdict(list)
    for element in live:
        if element.get('ID'):
            ids[element.get('ID')].append(element)

    duplicate_ids = {key: len(elems) for key, elems in ids.items()
                     if len(elems) != 1}
    connections = [e for e in live if local_name(e.tag) == 'Connection']
    endpoints = []

    for connection_index, connection in enumerate(connections):
        parent = parents.get(connection)
        relation = local_name(parent.tag) if parent is not None else None
        for side in ('From', 'To'):
            component_id = connection.get(side + 'ID')
            port_ref = connection.get(side + 'Node')
            candidates = ids.get(component_id, []) if component_id else []
            record = {
                'connection_index': connection_index,
                'connection_attributes': dict(connection.attrib),
                'parent_tag': relation,
                'parent_id': parent.get('ID') if parent is not None else None,
                'side': side,
                'component_id': component_id,
                'port_ref': port_ref,
                'status': None,
                'declared_nodes': [],
                'resolved_node': None,
                'node_type': None,
                'port_index_convention': 'zero_based_all_direct_Node_children',
                'type_consistency': 'not_checked',
            }
            if not component_id:
                record['status'] = 'missing_endpoint'
            elif len(candidates) == 0:
                record['status'] = 'unresolved_component_id'
            elif len(candidates) != 1:
                record['status'] = 'ambiguous_component_id'
            else:
                component = candidates[0]
                # Node order comes from XML; direct Node children only.
                nodes = [node for group in component
                         if local_name(group.tag) == 'ConnectionPoints'
                         for node in group if local_name(node.tag) == 'Node']
                record['declared_nodes'] = [dict(node.attrib) for node in nodes]

                if port_ref is None or not port_ref.strip():
                    record['status'] = 'missing_port_ref'
                elif not re.fullmatch(r'[0-9]+', port_ref):
                    record['status'] = 'invalid_port_ref_format'
                else:
                    index = int(port_ref)
                    if index >= len(nodes):
                        record['status'] = 'port_index_out_of_range'
                    else:
                        node = nodes[index]
                        record['resolved_node'] = dict(node.attrib)
                        record['node_type'] = node.get('Type')
                        record['status'] = 'index_resolved'
                        # Do not silently treat this as schema-level validity.
                        if relation == 'PipingNetworkSegment':
                            record['type_consistency'] = (
                                'process' if node.get('Type') == 'process'
                                else 'inspect_type')
                        elif relation == 'InformationFlow':
                            record['type_consistency'] = (
                                'signal' if node.get('Type') == 'signal'
                                else 'inspect_type')
            endpoints.append(record)

    # Extract evidence, NOT inferred physical direction. A PipeFlowArrow may
    # be nested under display/presentation elements, so inspect full ancestry.
    arrows = []
    for arrow in (e for e in live if local_name(e.tag) == 'PipeFlowArrow'):
        chain = parent_chain(arrow, parents)
        segment = next((a for a in chain if a['tag'] == 'PipingNetworkSegment'), None)
        # Keep nearby source XML rather than interpreting geometry transforms.
        descendants = list(arrow.iter())
        arrows.append({
            'id': arrow.get('ID'),
            'attributes': dict(arrow.attrib),
            'ancestors': chain,
            'containing_segment_id': segment['id'] if segment else None,
            'generic_attributes': own_generic_attributes(arrow),
            'descendants': [
                {'tag': local_name(e.tag), 'attributes': dict(e.attrib),
                 'text': (e.text or '').strip()[:300]}
                for e in descendants[:100]
            ],
            'descendants_truncated': len(descendants) > 100,
            'direction_status': 'not_inferred',
        })

    # Capture explicit direction-related metadata anywhere in live XML, so
    # generic attributes and direction flags are not overlooked.
    direction_hints = []
    direction_pattern = re.compile(r'(flow.?direction|flow.?in|flow.?out|pipe.?flow.?arrow)', re.I)
    for element in live:
        matches = {k: v for k, v in element.attrib.items()
                   if direction_pattern.search(k) or
                   (local_name(element.tag) == 'GenericAttribute'
                    and direction_pattern.search(str(element.get('Name', ''))))}
        if matches:
            direction_hints.append({
                'tag': local_name(element.tag),
                'id': element.get('ID'),
                'attributes': matches,
                'ancestors': parent_chain(element, parents),
            })

    return {
        'source_file': str(xml_path),
        'assumptions': {
            'port_indexing': 'zero-based among direct ConnectionPoints/Node children; confirm against applicable Proteus specification',
            'from_to_direction': 'connection endpoint ordering, NOT verified process flow',
            'flow_arrow_direction': 'not inferred from raw geometry',
            'flow_direction_availability': 'absence of evidence is not evidence of no flow',
        },
        'summary': {
            'shape_catalogue_elements_excluded': excluded_count,
            'connection_count': len(connections),
            'endpoint_count': len(endpoints),
            'endpoint_status_counts': dict(Counter(r['status'] for r in endpoints)),
            'endpoint_type_counts': dict(Counter(r['type_consistency'] for r in endpoints)),
            'duplicate_id_count': len(duplicate_ids),
            'pipe_flow_arrow_count': len(arrows),
            'direction_hint_count': len(direction_hints),
        },
        'duplicate_ids': duplicate_ids,
        'endpoint_results': endpoints,
        'flow_arrows': arrows,
        'direction_hints': direction_hints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xml', type=Path, help='DEXPI Proteus XML input')
    parser.add_argument('--output', type=Path, default=Path('port_flow_inspection.json'))
    args = parser.parse_args()
    if not args.xml.is_file():
        parser.error(f'Input XML not found: {args.xml}')
    report = inspect(args.xml)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
