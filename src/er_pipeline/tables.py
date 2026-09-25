"""Export final model tables and challenge-format final candidate lists."""
import csv
import json
import os
import shutil
from collections import Counter
from contextlib import ExitStack

import pyarrow.parquet as pq

from .common import ParquetSink, parquet_rows, rows, dump_json


def export(work, split, config):
    schema = pq.ParquetFile(work / 'pair_features.parquet').schema_arrow
    counts = Counter()
    if split == 'test':
        destination = work / 'test_features.parquet'
        if destination.exists():
            destination.unlink()
        try:
            os.link(work / 'pair_features.parquet', destination)
        except OSError:
            from .resources import check_disk
            check_disk(work, (work/'pair_features.parquet').stat().st_size)
            shutil.copyfile(work/'pair_features.parquet', destination)
        counts['test_pairs'] = pq.ParquetFile(destination).metadata.num_rows
    else:
        with ExitStack() as stack:
            groups = ('train', 'validation') if split == 'train' else ('test',)
            writers = {g: stack.enter_context(ParquetSink(work / f'{g}_features.parquet', schema, config['row_group_size'])) for g in groups}
            for pair in parquet_rows(work / 'pair_features.parquet'):
                group = pair['dataset_split'] if split == 'train' else 'test'
                writers[group].append(pair)
                counts[group + '_pairs'] += 1
                if split == 'train':
                    counts[group + '_positive_pairs'] += pair['label']
    # Keep zero-candidate queries: the pair table alone cannot represent them.
    pairs = iter(parquet_rows(work / 'candidate_pairs.parquet'))
    pair = next(pairs, None)
    with (work / 'candidate_pairs.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['source1_entity_id', 'candidate_entity_ids'])
        for query in rows(work / 'queries.tsv.gz'):
            sid, ids = query['source1_entity_id'], []
            while pair is not None and pair['source1_entity_id'] == sid:
                ids.append(pair['candidate_entity_id'])
                pair = next(pairs, None)
            if len(ids) != int(query['n_candidates']) or len(ids) != len(set(ids)):
                raise ValueError('Candidate/manifest inconsistency for ' + sid)
            writer.writerow([sid, ','.join(ids)])
            counts['queries'] += 1
        if pair is not None:
            raise ValueError('Unconsumed candidate rows: query order changed')
    if split == 'train':
        for query in rows(work / 'query_labels.tsv.gz'):
            counts[query['dataset_split'] + '_queries'] += 1
        for group in ('train', 'validation'):
            counts[group + '_negative_pairs'] = counts[group + '_pairs'] - counts[group + '_positive_pairs']
    dump_json(work / 'tables_report.json', dict(counts))
    print('Final feature tables and exact candidate lists exported.', flush=True)
