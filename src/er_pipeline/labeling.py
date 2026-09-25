"""Join candidates to truth, keep missed positives visible, assign S1-level splits."""
import csv
import json
from collections import defaultdict

from .common import (connect, rows, parquet_rows, truth_ids, split_for, dump_json,
                     ParquetSink, LABEL_SCHEMA, open_text)


def label(work, truth_path, config):
    cluster_path=work.parent/'clusters.sqlite'
    clusters=connect(cluster_path,readonly=True) if cluster_path.exists() else None
    path = work / 'truth.sqlite.partial'
    if path.exists():
        path.unlink()
    conn = connect(path, config['sqlite_cache_mb'])
    conn.execute('CREATE TABLE truth(sid TEXT PRIMARY KEY, ids TEXT NOT NULL) WITHOUT ROWID')
    for i, row in enumerate(rows(truth_path), 1):
        ids = truth_ids(row['matched_entity_ids'])
        conn.execute('INSERT INTO truth VALUES(?,?)', (row['source1_entity_id'], json.dumps(ids)))
        if i % 10000 == 0:
            conn.commit()
    conn.commit()
    conn.execute('CREATE TABLE queries(sid TEXT PRIMARY KEY, country TEXT, dataset_split TEXT, n_candidates INTEGER) WITHOUT ROWID')
    for row in rows(work / 'queries.tsv.gz'):
        value = conn.execute('SELECT ids FROM truth WHERE sid=?', (row['source1_entity_id'],)).fetchone()
        if value is None:
            raise ValueError('Missing ground-truth row: ' + row['source1_entity_id'])
        group = split_for(row['source1_entity_id'], row['country'], bool(json.loads(value[0])), config)
        if clusters:
            assigned=clusters.execute('SELECT split FROM queries WHERE sid=?',(row['source1_entity_id'],)).fetchone()
            if not assigned:raise ValueError('Missing cluster assignment')
            group=assigned[0]
        conn.execute('INSERT INTO queries VALUES(?,?,?,?)', (row['source1_entity_id'], row['country'], group, int(row['n_candidates'])))
    conn.commit()
    conn.execute('CREATE TABLE hits(sid TEXT PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID')
    refs = connect(work / 'index.sqlite', config['sqlite_cache_mb'], readonly=True)
    # Audit ground-truth IDs for selected queries. Missing references are input errors.
    for sid, value in conn.execute('SELECT sid,ids FROM truth JOIN queries USING(sid)'):
        for eid in json.loads(value):
            if not refs.execute('SELECT 1 FROM records WHERE entity_id=? AND source IN (2,3)', (eid,)).fetchone():
                raise ValueError(f'{sid}: labeled reference {eid} absent from indexed S2/S3')
    refs.close()
    if clusters:clusters.close()
    current, truth, group, hits = None, set(), None, 0
    with ParquetSink(work / 'labeled_pairs.parquet', LABEL_SCHEMA, config['row_group_size']) as sink:
        for pair_index, pair in enumerate(parquet_rows(work / 'candidate_pairs.parquet'), 1):
            if pair_index % 100000 == 0:
                conn.commit()
            sid = pair['source1_entity_id']
            if sid != current:
                if current is not None:
                    conn.execute('INSERT INTO hits VALUES(?,?)', (current, hits))
                value = conn.execute('SELECT ids,dataset_split FROM truth JOIN queries USING(sid) WHERE sid=?', (sid,)).fetchone()
                if value is None:
                    raise ValueError('Candidate query has no truth/split: ' + sid)
                truth, group, current, hits = set(json.loads(value[0])), value[1], sid, 0
            label_value = int(pair['candidate_entity_id'] in truth)
            hits += label_value
            sink.append(dict(pair, label=label_value, dataset_split=group))
        if current is not None:
            conn.execute('INSERT INTO hits VALUES(?,?)', (current, hits))
    conn.commit()
    report = defaultdict(lambda: dict(queries=0, true_links=0, retrieved_true_links=0,
        queries_with_matches=0, queries_all_matches_retrieved=0, singleton_queries=0,
        candidate_pairs=0, oracle_f05_sum=0.0))
    fields = ['source1_entity_id', 'country', 'dataset_split', 'matched_entity_ids',
              'n_candidates', 'true_links', 'retrieved_true_links']
    with open_text(work / 'query_labels.tsv.gz', 'wt') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        sql = '''SELECT q.sid,q.country,q.dataset_split,q.n_candidates,t.ids,COALESCE(h.n,0)
                 FROM queries q JOIN truth t USING(sid) LEFT JOIN hits h USING(sid) ORDER BY q.sid'''
        for sid, country, group, pairs, value, hit in conn.execute(sql):
            ids = json.loads(value)
            n = len(ids)
            writer.writerow(dict(source1_entity_id=sid, country=country, dataset_split=group,
                matched_entity_ids=','.join(ids), n_candidates=pairs, true_links=n, retrieved_true_links=hit))
            for key in ('all', group, 'country:' + country, group + ':' + country):
                stats = report[key]
                stats['queries'] += 1
                stats['true_links'] += n
                stats['retrieved_true_links'] += hit
                stats['queries_with_matches'] += n > 0
                stats['queries_all_matches_retrieved'] += n > 0 and n == hit
                stats['singleton_queries'] += n == 0
                stats['candidate_pairs'] += pairs
                stats['oracle_f05_sum'] += 1.0 if n == 0 else 5*hit / (4*hit+n)
    for stats in report.values():
        stats['candidate_recall'] = stats['retrieved_true_links'] / stats['true_links'] if stats['true_links'] else None
        stats['all_matches_retrieved_rate'] = stats['queries_all_matches_retrieved'] / stats['queries_with_matches'] if stats['queries_with_matches'] else None
        stats['oracle_macro_f05_ceiling'] = stats.pop('oracle_f05_sum') / stats['queries']
        stats['positive_pairs'] = stats['retrieved_true_links']
        stats['negative_pairs'] = stats['candidate_pairs'] - stats['positive_pairs']
    dump_json(work / 'candidate_recall.json', dict(report))
    conn.close()
    path.replace(work / 'truth.sqlite')
    print('Labeling complete. Recall includes missed true matches; none were injected.', flush=True)
