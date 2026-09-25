"""Shared streaming I/O, schemas and configuration."""
import csv
import gzip
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULTS = dict(hash_bits=18, validation_fraction=0.2, seed=42,
    block_limit=100, lexical_pool=120, lexical_top_k=25,
    max_candidates=80, query_terms=16, row_group_size=10000,
    sqlite_cache_mb=128, progress_every=10000, india_retrieval_multiplier=1,
    word_retrieval=1, diverse_candidates=8)
PAIR_FIELDS = ['source1_entity_id', 'candidate_entity_id', 'candidate_source',
    'candidate_rank', 'retrieval_score', 'name_retrieval_cosine',
    'address_retrieval_cosine', 'block_A', 'block_B', 'block_C', 'block_D',
    'block_E', 'block_F', 'block_G', 'neural_cosine', 'neural_rank']
PAIR_SCHEMA = pa.schema([(k, pa.string() if k.endswith('entity_id') else
    pa.float32() if 'score' in k or 'cosine' in k else pa.int32()) for k in PAIR_FIELDS])
LABEL_SCHEMA = pa.schema(list(PAIR_SCHEMA) + [pa.field('label', pa.int8()), pa.field('dataset_split', pa.string())])


def open_text(path, mode='rt'):
    return (gzip.open if str(path).endswith('.gz') else open)(path, mode, encoding='utf-8-sig' if 'r' in mode else 'utf-8', newline='')


def rows(path):
    with open_text(path) as f:
        yield from csv.DictReader(f, delimiter='\t')


def parquet_rows(path, columns=None):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


class ParquetSink:
    def __init__(self, path, schema, chunk=10000):
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + '.partial')
        from .resources import batch_rows, check_disk
        check_disk(self.path.parent)
        self.schema, self.chunk, self.buffer = schema, batch_rows(chunk, len(schema)), []
        self.writer = pq.ParquetWriter(self.partial, schema, compression='zstd')
    def append(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= self.chunk:
            self.flush()
    def flush(self):
        if self.buffer:
            from .resources import check_disk
            check_disk(self.path.parent)
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()
    def __enter__(self):
        return self
    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.flush()
        finally:
            self.writer.close()
        if kind is None:
            self.partial.replace(self.path)


def dump_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def connect(path, cache_mb=128, readonly=False):
    if readonly:
        conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=120)
    else:
        conn = sqlite3.connect(path, timeout=120)
    conn.execute(f'PRAGMA cache_size={-1024 * cache_mb}')
    conn.execute('PRAGMA temp_store=FILE')
    return conn


def find_source(root, split, number):
    hits = list(Path(root).rglob(f'{split}_source{number}.tsv.gz'))
    hits += list(Path(root).rglob(f'{split}_source{number}.tsv'))
    if len(hits) != 1:
        raise ValueError(f'Expected exactly one cleaned {split}_source{number}.tsv[.gz], found {len(hits)}')
    return hits[0]


def find_truth(root):
    hits = list(Path(root).rglob('train_ground_truth.tsv'))
    if len(hits) != 1:
        raise ValueError('Expected exactly one train_ground_truth.tsv')
    return hits[0]


def split_for(sid, country, has_match, config):
    # Stable approximate stratification, NOT a random pair-level split.
    key = f'{config["seed"]}|{country}|{int(has_match)}|{sid}'.encode()
    u = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big') / 2**64
    return 'validation' if u < config['validation_fraction'] else 'train'


def lookup(conn, eid):
    row = conn.execute('SELECT payload FROM records WHERE entity_id=?', (eid,)).fetchone()
    if row is None:
        raise ValueError(f'Unknown record: {eid}')
    return json.loads(row[0])


def truth_ids(raw):
    ids = [s.strip() for s in raw.split(',') if s.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate matched ID in ground truth')
    if any(not x.startswith(('S2-', 'S3-')) for x in ids):
        raise ValueError('Ground truth contains a non-S2/S3 ID')
    return ids
