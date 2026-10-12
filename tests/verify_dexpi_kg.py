"""Read-only DEXPI/Proteus source audit and optional LightRAG GraphML audit.

uv run python verify_dexpi_kg.py --dexpi data/others/C01V04-VER.EX01.xml
Add --graph PATH --chunks PATH [--mapping PATH] to audit stored extraction.
Mapping JSON: {"XML-ID": "exact graph node ID"}. No LLM calls are made.

Scope: explicit segment endpoints, membership, equipment ownership, numeric
source attributes, and local endpoint FlowIn/FlowOut markers. This is not a
complete Proteus interpreter: no geometry-based ordering, process simulation,
internal nozzle-to-nozzle traversal, or automatic semantic fact scoring.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

OBJECT_TAGS = {'Equipment', 'Nozzle', 'PipingComponent', 'PipeOffPageConnector',
               'PipingNetworkSystem', 'PipingNetworkSegment',
               'ProcessInstrumentationFunction', 'InformationFlow',
               'ActuatingSystem', 'ActuatingFunction', 'ActuatingSystemComponent',
               'ProcessSignalGeneratingFunction', 'InstrumentationLoopFunction'}
TOPOLOGY = {'connected_to', 'upstream_of', 'downstream_of', 'up_stream_of',
            'down_stream_of', 'feeds', 'discharges_to'}
MEMBERSHIP = {'contains', 'part_of', 'belongs_to', 'contained_in'}
OWNER = {'has_parameter', 'has_limit'}


def local(tag):
    return tag.rsplit('}', 1)[-1]


def children(element, tag):
    return [c for c in element if local(c.tag) == tag]


def attributes(element):
    return [dict(a.attrib) for group in children(element, 'GenericAttributes')
            for a in children(group, 'GenericAttribute')]


def value(record, name):
    return next((a.get('Value', '') for a in record['attributes']
                 if a.get('Name') == name), '')


def source_model(path):
    data = path.read_bytes()
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise ValueError('DTD/entity declarations are not supported')
    root = ET.fromstring(data)
    parents = {c: p for p in root.iter() for c in p}

    def in_catalogue(e):
        while e in parents:
            e = parents[e]
            if local(e.tag) == 'ShapeCatalogue':
                return True
        return False

    objects = {}
    elements = {}
    excluded = 0
    for e in root.iter():
        if local(e.tag) not in OBJECT_TAGS or not e.get('ID'):
            continue
        if in_catalogue(e):
            excluded += 1
            continue
        identifier = e.get('ID')
        if identifier in objects:
            raise ValueError(f'Duplicate XML object ID: {identifier}')
        cp = children(e, 'ConnectionPoints')
        objects[identifier] = {
            'id': identifier, 'xml_tag': local(e.tag),
            'component_class': e.get('ComponentClass', ''),
            'attributes': attributes(e),
            'flow_markers': dict(cp[0].attrib) if cp else {},
            'ports': [dict(n.attrib) for n in children(cp[0], 'Node')] if cp else [],
            'associations': [dict(a.attrib) for a in children(e, 'Association')],
        }
        elements[identifier] = e
    for identifier, e in elements.items():
        p = parents.get(e)
        while p is not None and p.get('ID') not in objects:
            p = parents.get(p)
        objects[identifier]['parent_id'] = p.get('ID') if p is not None else None
        objects[identifier]['tag_name'] = value(objects[identifier], 'TagNameAssignmentClass')

    connections = []
    issues = []
    for identifier, e in elements.items():
        if local(e.tag) not in {'PipingNetworkSegment', 'InformationFlow'}:
            continue
        for c in children(e, 'Connection'):
            row = {'owner_id': identifier, 'kind': 'process' if local(e.tag) == 'PipingNetworkSegment' else 'signal',
                   'xml_attributes': dict(c.attrib), 'endpoints': []}
            for role, key, port_key in [('source', 'FromID', 'FromNode'), ('target', 'ToID', 'ToNode')]:
                item = c.get(key)
                if not item:
                    continue
                endpoint = {'role': role, 'item_id': item, 'port_index': c.get(port_key),
                            'local_flow': 'unknown'}
                if item not in objects:
                    issues.append({'kind': 'unresolved_endpoint', 'owner': identifier, 'item': item})
                else:
                    obj = objects[item]
                    port = c.get(port_key)
                    if port is not None:
                        try:
                            index = int(port)
                            if index < 0 or index >= len(obj['ports']):
                                raise ValueError()
                            endpoint['port_id'] = obj['ports'][index].get('ID')
                            if row['kind'] == 'process':
                                markers = obj['flow_markers']
                                inflow = markers.get('FlowIn') == port
                                outflow = markers.get('FlowOut') == port
                                endpoint['local_flow'] = ('conflicting' if inflow and outflow else
                                                          'into_item' if inflow else
                                                          'out_of_item' if outflow else 'unknown')
                        except ValueError:
                            issues.append({'kind': 'invalid_port_index', 'owner': identifier,
                                           'item': item, 'index': port})
                row['endpoints'].append(endpoint)
            connections.append(row)
    numeric = [{'owner_id': identifier, **a} for identifier, obj in objects.items()
               for a in obj['attributes'] if a.get('Format') in {'double', 'float', 'integer', 'int', 'decimal'}
               and a.get('Value') is not None]
    return {
        'source_file': path.name,
        'plant_information': next((dict(e.attrib) for e in root if local(e.tag) == 'PlantInformation'), {}),
        'counts': dict(Counter(o['xml_tag'] for o in objects.values())),
        'excluded_catalogue_objects': excluded,
        'objects': objects, 'connections': connections,
        'numeric_facts': numeric, 'source_issues': issues,
        'limitations': [
            'Source/target endpoint roles are not automatically process flow direction.',
            'A segment can contain inline components; its endpoints are not necessarily directly adjacent.',
            'Local FlowIn/FlowOut evidence does not establish a complete equipment-level flow path.',
            'Signal connections are excluded from process-direction checks.',
            'ShapeCatalogue definitions are excluded. No geometry or internal equipment flow is inferred.',
        ],
    }


def tokens(attrs):
    return set(re.findall(r'[a-z][a-z0-9_]*', str(attrs.get('keywords', '')).lower()))


def graph_edges(graph):
    if graph.is_multigraph():
        return list(graph.edges(keys=True, data=True))
    return [(u, v, None, a) for u, v, a in graph.edges(data=True)]


def evidence_text(candidate):
    """Search factual attributes, excluding identifiers and provenance metadata."""
    ignored = {'source_id', 'file_path', 'full_doc_id', 'entity_id', 'dexpi_id',
               'id', 'tag', 'weight', 'created_at', 'timestamp'}
    parts = [str(v) for k, v in candidate.get('attributes', {}).items() if k not in ignored]
    parts.extend(str(v) for k, v in candidate.get('edge', {}).items()
                 if k not in ignored | {'u', 'v', 'key'})
    return ' '.join(parts)


def audit(model, graph_path, chunks_path, mapping_path):
    import networkx as nx
    g = nx.read_graphml(graph_path)
    chunks = json.loads(chunks_path.read_text(encoding='utf-8-sig')) if chunks_path else None
    if chunks is not None and (not isinstance(chunks, dict) or
                               any(not isinstance(v, dict) for v in chunks.values())):
        raise ValueError('Chunks must be a JSON object keyed by chunk ID with object values')
    manual = json.loads(mapping_path.read_text(encoding='utf-8-sig')) if mapping_path else {}
    if not isinstance(manual, dict):
        raise ValueError('Mapping must be a JSON object')
    for source_id, node in manual.items():
        if source_id not in model['objects'] or node not in g:
            raise ValueError(f'Invalid mapping: {source_id!r} -> {node!r}')
    mapped = {}
    for identifier, obj in model['objects'].items():
        aliases = {identifier}
        if obj['tag_name']:
            aliases.add(obj['tag_name'])
        matches = []
        for node, attrs in g.nodes(data=True):
            labels = {str(node).casefold(), str(attrs.get('entity_id', '')).casefold(),
                      str(attrs.get('dexpi_id', '')).casefold(), str(attrs.get('tag', '')).casefold()}
            if labels & {a.casefold() for a in aliases}:
                matches.append(node)
        if identifier in manual:
            mapped[identifier] = {'status': 'manual', 'node': manual[identifier]}
        elif len(matches) == 1:
            mapped[identifier] = {'status': 'exact_identifier_or_tag', 'node': matches[0]}
        else:
            # Description mentions are hints only, not identity confirmation.
            hints = [str(n) for n, a in g.nodes(data=True)
                     if any(re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)',
                                      str(n) + ' ' + str(a.get('description', '')), re.I)
                            for alias in aliases)]
            mapped[identifier] = {'status': 'ambiguous' if matches else 'unmapped',
                                  'exact_candidates': matches, 'review_candidates': hints[:20]}
    inverse = defaultdict(list)
    for identifier, m in mapped.items():
        if 'node' in m:
            inverse[m['node']].append(identifier)
    edge_list = graph_edges(g)
    checks = []

    def between(a, b):
        return [(u, v, k, attrs) for u, v, k, attrs in edge_list
                if (u == a and v == b) or (u == b and v == a)]

    for connection in model['connections']:
        for endpoint in connection['endpoints']:
            row = {'kind': connection['kind'], 'segment_or_signal_id': connection['owner_id'], **endpoint}
            a = mapped.get(connection['owner_id'], {}).get('node')
            b = mapped.get(endpoint['item_id'], {}).get('node')
            if a is None or b is None:
                row['status'] = 'unmapped'
            else:
                candidates = between(a, b)
                row['edges'] = [{'u': u, 'v': v, 'key': key, **attrs} for u, v, key, attrs in candidates]
                row['status'] = ('topology_keyword_present_review_semantics' if
                                 any(tokens(attrs) & TOPOLOGY for _, _, _, attrs in candidates)
                                 else 'other_edge_only' if candidates else 'endpoint_segment_edge_not_found')
                row['direction_status'] = ('manual_semantic_review_required' if endpoint['local_flow'] != 'unknown'
                                           else 'no_explicit_local_flow_marker')
                # Compare only local item/segment claims. Never propagate a
                # direction through equipment or use undirected edge order.
                claims = []
                expected = ((b, a) if endpoint['local_flow'] == 'out_of_item' else
                            (a, b) if endpoint['local_flow'] == 'into_item' else None)
                for u, v, key, attrs in candidates:
                    for keyword in tokens(attrs) & {'upstream_of', 'up_stream_of',
                                                    'downstream_of', 'down_stream_of',
                                                    'feeds', 'discharges_to'}:
                        claimed = (v, u) if keyword in {'downstream_of', 'down_stream_of'} else (u, v)
                        status = ('direction_unverifiable_in_undirected_graph' if not g.is_directed() else
                                  'no_unambiguous_source_local_direction' if expected is None else
                                  'matches_explicit_local_flow_marker' if claimed == expected else
                                  'contradicts_explicit_local_flow_marker')
                        claims.append({'keyword': keyword, 'edge_key': key, 'status': status,
                                       'claimed_flow': list(claimed) if g.is_directed() else None,
                                       'expected_local_flow': list(expected) if expected else None})
                row['local_direction_claim_checks'] = claims
            checks.append(row)
    memberships = []
    for identifier, obj in model['objects'].items():
        parent = obj['parent_id']
        if parent not in mapped:
            continue
        a, b = mapped[identifier].get('node'), mapped[parent].get('node')
        candidates = between(a, b) if a is not None and b is not None else []
        memberships.append({'item': identifier, 'parent': parent,
                            'status': 'unmapped' if a is None or b is None else
                            'membership_keyword_present_review_semantics' if
                            any(tokens(attrs) & MEMBERSHIP for _, _, _, attrs in candidates) else
                            'membership_edge_not_found'})
    facts = []
    for fact in model['numeric_facts']:
        node = mapped[fact['owner_id']].get('node')
        candidates = []
        if node is not None:
            candidates.append({'node': node, 'attributes': dict(g.nodes[node])})
            for u, v, key, attrs in edge_list:
                if node in (u, v) and tokens(attrs) & OWNER:
                    other = v if u == node else u
                    candidates.append({'node': other, 'attributes': dict(g.nodes[other]),
                                       'edge': {'u': u, 'v': v, 'key': key, **attrs}})
        # Literal matches only: never assign a semantic accuracy score.
        pattern = r'(?<![\w.])' + re.escape(fact['Value']) + r'(?![\w.])'
        hits = [c for c in candidates if re.search(pattern, evidence_text(c))]
        facts.append({**fact, 'status': 'unmapped_owner' if node is None else
                      'value_mention_found_requires_parameter_and_unit_review' if hits else
                      'exact_value_mention_not_found', 'candidate_evidence': hits})
    provenance = []
    for kind, records in [('node', [(str(n), a) for n, a in g.nodes(data=True)]),
                          ('edge', [(f'{u} -- {v} [{key}]', a) for u, v, key, a in edge_list])]:
        for identifier, attrs in records:
            refs = [s.strip() for s in str(attrs.get('source_id', '')).split('<SEP>') if s.strip()]
            unresolved = [s for s in refs if chunks is not None and s not in chunks]
            if not refs or unresolved:
                provenance.append({'kind': kind, 'id': identifier, 'source_ids': refs,
                                   'unresolved_ids': unresolved,
                                   'status': 'missing_source_id' if not refs else 'unresolved_in_chunks_review_multimodal_refs'})
    chunk_review = []
    if chunks is not None:
        for chunk_id, chunk in chunks.items():
            content = str(chunk.get('content', ''))
            ids = [identifier for identifier in model['objects']
                   if re.search(r'(?<!\w)' + re.escape(identifier) + r'(?!\w)', content)]
            chunk_review.append({'chunk_id': chunk_id, 'file_path': chunk.get('file_path'),
                                 'full_doc_id': chunk.get('full_doc_id'), 'dexpi_id_mentions': ids,
                                 'content_characters': len(content)})
    cross_document = []
    installation = {'installed_on', 'installed_in', 'monitored_by', 'controlled_by', 'protected_by'}
    for u, v, key, attrs in edge_list:
        if bool(inverse.get(u)) != bool(inverse.get(v)) and tokens(attrs) & installation:
            cross_document.append({'u': u, 'v': v, 'key': key, **attrs,
                                   'status': 'review_external_relationship_source_evidence'})
    topology_review = [
        {'u': u, 'v': v, 'key': key, **attrs,
         'legacy_direction_spelling': sorted(tokens(attrs) & {'up_stream_of', 'down_stream_of'}),
         'status': 'review_source_and_endpoint_checks_no_global_path_validation'}
        for u, v, key, attrs in edge_list if tokens(attrs) & TOPOLOGY
    ]
    return {
        'graph_nodes': g.number_of_nodes(), 'graph_edges': g.number_of_edges(),
        'directed': g.is_directed(),
        'direction_warning': 'Undirected edge endpoint order cannot establish subject/object direction.' if not g.is_directed() else
                             'Graph direction alone does not establish upstream/downstream semantics.',
        'mapping': mapped, 'mapping_counts': dict(Counter(m['status'] for m in mapped.values())),
        'possible_merges': {str(n): ids for n, ids in inverse.items() if len(ids) > 1},
        'endpoint_segment_checks': checks, 'membership_checks': memberships,
        'numeric_fact_review': facts, 'provenance_flags': provenance,
        'chunks': chunk_review, 'external_installation_relationship_review': cross_document,
        'all_topology_edges_for_review': topology_review,
        'limits': 'This audit finds representation gaps and review evidence, not semantic truth. '
                  'Alternate KG schemas may encode connections differently. Unit conversions and aliases need review. '
                  'Cross-document identity needs an explicit tag/model mapping or other source evidence.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dexpi', required=True, type=Path)
    p.add_argument('--graph', type=Path)
    p.add_argument('--chunks', type=Path)
    p.add_argument('--mapping', type=Path)
    p.add_argument('--output', type=Path, default=Path('dexpi_verification.json'))
    args = p.parse_args()
    if (args.chunks or args.mapping) and not args.graph:
        p.error('--chunks and --mapping require --graph')
    inputs = [x.resolve() for x in (args.dexpi, args.graph, args.chunks, args.mapping) if x]
    if args.output.resolve() in inputs:
        p.error('Output must not overwrite any input')
    try:
        model = source_model(args.dexpi)
        report = {'source_reference': model}
        if args.graph:
            report['kg_audit'] = audit(model, args.graph, args.chunks, args.mapping)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    except (OSError, ValueError, ET.ParseError, ImportError) as exc:
        p.error(str(exc))
    print('SOURCE COUNTS:', json.dumps(model['counts']))
    print('CONNECTIONS:', json.dumps(dict(Counter(c['kind'] for c in model['connections']))))
    print('NUMERIC SOURCE FACTS:', len(model['numeric_facts']))
    print('SOURCE ISSUES:', len(model['source_issues']))
    if 'kg_audit' in report:
        result = report['kg_audit']
        print('KG:', result['graph_nodes'], 'nodes /', result['graph_edges'], 'edges')
        print('MAPPING:', json.dumps(result['mapping_counts']))
        print('ENDPOINT CHECKS:', json.dumps(dict(Counter(c['status'] for c in result['endpoint_segment_checks']))))
        print('PROVENANCE FLAGS:', len(result['provenance_flags']))
    else:
        print('KG NOT CHECKED: add --graph and --chunks to compare ingestion/extraction.')
    print('No automatic semantic accuracy score; review the reported evidence.')
    print('REPORT:', args.output.resolve())


if __name__ == '__main__':
    main()
