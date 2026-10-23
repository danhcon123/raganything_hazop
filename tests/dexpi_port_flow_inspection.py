"""Enrich a DEXPI port-flow inspection report with human-readable component metadata.

Usage:
    python dexpi_inspection_enrich.py --xml C01V04-VER.EX01.xml \
        --report port_flow_inspection.json --output-dir inspection_enriched

Only reads DEXPI XML and an existing inspection report. No LLM or GraphML required.
"""
from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def direct_child(element: ET.Element, name: str) -> ET.Element | None:
    return next((c for c in element if local(c.tag) == name), None)


def direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in element if local(c.tag) == name]


def generic_attributes(element: ET.Element) -> dict[str, list[str]]:
    attrs: dict[str, list[str]] = defaultdict(list)
    for group in direct_children(element, 'GenericAttributes'):
        for attr in direct_children(group, 'GenericAttribute'):
            if attr.get('Value') is not None:
                attrs[attr.get('Name', '')].append(attr.get('Value', ''))
    return dict(attrs)


def first(attrs: dict[str, list[str]], name: str) -> str:
    return next(iter(attrs.get(name, [])), '')


def own_position(element: ET.Element) -> dict[str, str] | None:
    position = direct_child(element, 'Position')
    loc = direct_child(position, 'Location') if position is not None else None
    return dict(loc.attrib) if loc is not None else None


def own_labels(element: ET.Element) -> list[str]:
    labels = []
    for label in direct_children(element, 'Label'):
        for child in label.iter():
            if local(child.tag) == 'Text' and child.get('String'):
                labels.append(child.get('String', ''))
    return list(dict.fromkeys(labels))


def export_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def enrich(xml_path: Path, report_path: Path, output_dir: Path) -> dict:
    raw = xml_path.read_bytes()
    if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('DTD and entity declarations are not accepted')
    root = ET.fromstring(raw)
    parents = {child: parent for parent in root.iter() for child in parent}
    # All live IDs; catalogue entries are intentionally excluded.
    live = []
    for element in root.iter():
        if not element.get('ID'):
            continue
        cur = element
        in_catalogue = False
        while cur in parents:
            cur = parents[cur]
            if local(cur.tag) == 'ShapeCatalogue':
                in_catalogue = True
                break
        if not in_catalogue:
            live.append(element)
    index = {}
    for element in live:
        identifier = element.get('ID')
        if identifier in index:
            raise ValueError(f'Duplicate live XML ID: {identifier}')
        index[identifier] = element

    metadata = {}
    for identifier, element in index.items():
        attrs = generic_attributes(element)
        parent = parents.get(element)
        owners = {}
        while parent is not None:
            tag = local(parent.tag)
            if tag in ('Equipment', 'PipingNetworkSegment', 'PipingNetworkSystem') and tag not in owners:
                owners[tag] = parent.get('ID', '')
            parent = parents.get(parent)
        position = own_position(element)
        labels = own_labels(element)
        name = (first(attrs, 'TagNameAssignmentClass') or
                first(attrs, 'PipingComponentNameAssignmentClass') or
                first(attrs, 'PositionNumberAssignmentClass') or
                first(attrs, 'ActuatingSystemNumberAssignmentClass') or
                first(attrs, 'SubTagNameAssignmentClass'))
        metadata[identifier] = {
            'xml_id': identifier,
            'xml_type': local(element.tag),
            'component_class': element.get('ComponentClass', ''),
            'engineering_name': name,
            'display_labels': labels,
            'component_number': first(attrs, 'PipingComponentNumberAssignmentClass'),
            'line_number': first(attrs, 'LineNumberAssignmentClass'),
            'parent_equipment': owners.get('Equipment', ''),
            'parent_segment': owners.get('PipingNetworkSegment', ''),
            'parent_system': owners.get('PipingNetworkSystem', ''),
            'x': position.get('X', '') if position else '',
            'y': position.get('Y', '') if position else '',
            'position_present': position is not None,
        }
    # Inherit line number from the owning piping network system for review convenience.
    for item in metadata.values():
        if not item['line_number'] and item['parent_system'] in metadata:
            item['line_number'] = metadata[item['parent_system']]['line_number']

    with report_path.open(encoding='utf-8-sig') as f:
        report = json.load(f)
    endpoint_results = report.get('endpoint_results', [])
    by_conn: dict[int, dict] = {}
    for item in endpoint_results:
        n = item['connection_index']
        row = by_conn.setdefault(n, {
            'connection_index': n,
            'kind': 'process' if item.get('parent_tag') == 'PipingNetworkSegment' else 'signal',
            'owner_id': item.get('parent_id', ''),
            'from_id': item.get('connection_attributes', {}).get('FromID', ''),
            'from_node': item.get('connection_attributes', {}).get('FromNode', ''),
            'to_id': item.get('connection_attributes', {}).get('ToID', ''),
            'to_node': item.get('connection_attributes', {}).get('ToNode', ''),
        })
        prefix = 'from' if item.get('side') == 'From' else 'to'
        row[f'{prefix}_port_id'] = (item.get('resolved_node') or {}).get('ID', '')
        row[f'{prefix}_status'] = item.get('status', '')
        row[f'{prefix}_port_type'] = item.get('node_type') or ''

    connections = []
    for n, row in sorted(by_conn.items()):
        for prefix in ('from', 'to'):
            meta = metadata.get(row[f'{prefix}_id'], {})
            for field in ('component_class', 'engineering_name', 'display_labels', 'parent_equipment', 'x', 'y'):
                v = meta.get(field, '')
                row[f'{prefix}_{field}'] = ' | '.join(v) if isinstance(v, list) else v
        owner = metadata.get(row['owner_id'], {})
        row['line_number'] = owner.get('line_number', '')
        row['owner_parent_system'] = owner.get('parent_system', '')
        row['direction_status'] = 'unknown; FromID/ToID are not verified process flow'
        connections.append(row)

    port_uses = defaultdict(list)
    for row in connections:
        if row['kind'] != 'process':
            continue
        for prefix in ('from', 'to'):
            if row[f'{prefix}_id'] and row.get(f'{prefix}_port_id'):
                port_uses[(row[f'{prefix}_id'], row[f'{prefix}_port_id'])].append(row['connection_index'])
    reuse_flags = [{'component_id': obj, 'port_id': port, 'connection_indices': indices}
                   for (obj, port), indices in sorted(port_uses.items()) if len(indices) > 1]

    arrows = []
    for arrow in report.get('flow_arrows', []):
        position = next((x['attributes'] for x in arrow.get('descendants', [])
                         if x.get('tag') == 'Location'), {})
        arrows.append({
            'arrow_id': arrow.get('id', ''),
            'segment_id': arrow.get('containing_segment_id', ''),
            'line_number': metadata.get(arrow.get('containing_segment_id', ''), {}).get('line_number', ''),
            'x': position.get('X', ''), 'y': position.get('Y', ''),
            'direction_status': 'not_inferred',
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    component_rows = list(metadata.values())
    export_csv(output_dir / 'dexpi_component_index.csv', component_rows,
               ['xml_id','xml_type','component_class','engineering_name','display_labels',
                'component_number','line_number','parent_equipment','parent_segment',
                'parent_system','x','y','position_present'])
    export_csv(output_dir / 'dexpi_connection_review.csv', connections,
               ['connection_index','kind','owner_id','owner_parent_system','line_number',
                'from_id','from_component_class','from_engineering_name','from_display_labels',
                'from_parent_equipment','from_node','from_port_id','from_status','from_port_type','from_x','from_y',
                'to_id','to_component_class','to_engineering_name','to_display_labels',
                'to_parent_equipment','to_node','to_port_id','to_status','to_port_type','to_x','to_y',
                'direction_status'])
    export_csv(output_dir / 'dexpi_flow_arrows.csv', arrows,
               ['arrow_id','segment_id','line_number','x','y','direction_status'])
    enriched = {
        'source_xml': str(xml_path), 'source_report': str(report_path),
        'components': metadata, 'connections': connections,
        'flow_arrows': arrows, 'process_port_reuse_review': reuse_flags,
        'limitations': ['Port indices inherit the original report convention, which needs specification validation.',
                        'No process flow direction is inferred from FromID/ToID or geometry.',
                        'Display labels can be absent; engineering names and component numbers are distinct.',
                        'Component positions require a matching graphical P&ID for visual confirmation.'],
    }
    (output_dir / 'dexpi_enriched_inspection.json').write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'components': len(metadata), 'connections': len(connections),
            'flow_arrows': len(arrows), 'port_reuse_flags': len(reuse_flags),
            'output_dir': str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(enrich(args.xml, args.report, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
