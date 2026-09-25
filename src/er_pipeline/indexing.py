"""Build a disk-backed reference index. Store Source 1; index only S2 and S3."""
import json
import sqlite3
from pathlib import Path

from .common import connect, find_source, rows, dump_json
from .text_features import Tfidf, block_keys, lexical_tokens, name_text, address_text, encoded

FIELDS = ['entity_id', 'country_key', 'name_basic', 'name_expanded',
    'name_without_suffix', 'name_latin_folded', 'address_normalized',
    'address_latin_folded', 'postal_code_candidate', 'state_candidate',
    'city_candidate', 'house_number_zero_normalized', 'name_missing', 'address_missing']


def build(cleaned, work, split, config):
    path = work / 'index.sqlite.partial'
    if path.exists():
        path.unlink()  # Only this stage's incomplete output, never user input.
    conn = connect(path, config['sqlite_cache_mb'])
    conn.executescript('''
        CREATE TABLE records(rid INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL,
                             source INTEGER NOT NULL, payload TEXT NOT NULL);
        CREATE INDEX source_index ON records(source, rid);
        CREATE VIRTUAL TABLE search USING fts5(country, blocks, name_terms, address_terms,
                                content='', detail='column', tokenize='ascii');
    ''')
    tfidf = Tfidf(config['hash_bits']) if split == 'train' else None
    counts = {}
    try:
        for source in (1, 2, 3):
            count = 0
            for r in rows(find_source(cleaned, split, source)):
                missing = set(FIELDS) - r.keys()
                if missing:
                    raise ValueError(f'Run the supplied cleaner first; missing columns: {sorted(missing)}')
                record = {k: r[k] for k in FIELDS}
                record['source'] = source
                if not record['entity_id'].startswith(f'S{source}-'):
                    raise ValueError('Unexpected entity ID prefix: ' + record['entity_id'])
                for flag in ('name_missing', 'address_missing'):
                    record[flag] = int(record[flag])
                cur = conn.execute('INSERT INTO records(entity_id,source,payload) VALUES(?,?,?)',
                    (record['entity_id'], source, json.dumps(record, ensure_ascii=False)))
                if source != 1:
                    keys = block_keys(record)
                    conn.execute('INSERT INTO search(rowid,country,blocks,name_terms,address_terms) VALUES(?,?,?,?,?)',
                        (cur.lastrowid, encoded(record['country_key'] or 'UNKNOWN'),
                         ' '.join(k for group in keys.values() for k in group),
                         lexical_tokens(name_text(record)), lexical_tokens(address_text(record))))
                    if tfidf:
                        tfidf.update(record)
                count += 1
                if count % 10000 == 0:
                    from .resources import check_disk
                    check_disk(work)
                    conn.commit()
                if count % config['progress_every'] == 0:
                    print(f'index {split} S{source}: {count:,}', flush=True)
            counts[f'S{source}'] = count
            conn.commit()
        if not counts['S1'] or not (counts['S2'] + counts['S3']):
            raise ValueError('Need Source 1 queries and at least one reference record')
        print('Consolidating the search index...', flush=True)
        conn.execute("INSERT INTO search(search) VALUES('optimize')")
        conn.commit()
        conn.execute("CREATE INDEX country_source ON records(source,json_extract(payload,'$.country_key'),rid)")
        conn.commit()
        if tfidf:
            tfidf.save(work / 'tfidf.npz')
    finally:
        conn.close()
    path.replace(work / 'index.sqlite')
    dump_json(work / 'index_report.json', {'counts': counts, 'index_bytes': (work/'index.sqlite').stat().st_size,
        'tfidf_fit': 'train S2/S3 text only; held-out S1 text and all labels excluded'})
