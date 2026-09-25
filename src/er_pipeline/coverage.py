"""Check output S1 IDs against ORIGINAL raw test input, using constant memory."""
import csv
import io
import json
import zipfile
from contextlib import contextmanager
from itertools import zip_longest
from pathlib import Path

from .common import rows, dump_json


@contextmanager
def raw_source(path, filename):
    path = Path(path)
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            if any(Path(n).name == 'SAMPLE_README.txt' for n in archive.namelist()):
                raise ValueError('Known sample archive refused; provide the ORIGINAL challenge dataset')
            names = [n for n in archive.namelist() if Path(n).name == filename]
            if len(names) != 1:
                raise ValueError(f'Expected one {filename} in original archive')
            with archive.open(names[0]) as binary, io.TextIOWrapper(binary,encoding='utf-8-sig',newline='') as text:
                yield csv.DictReader(text,delimiter='\t')
    else:
        if any(path.rglob('SAMPLE_README.txt')):
            raise ValueError('Known sample directory refused')
        names = list(path.rglob(filename))
        if len(names) != 1:
            raise ValueError(f'Expected one {filename} in original directory')
        yield rows(names[0])


def validate(root, original):
    work = Path(root)/'test'
    # The pipeline preserves raw S1 order. Strict order equality implies full ID
    # coverage without keeping millions of IDs in a Python set.
    with raw_source(original,'test_source1.tsv') as official:
        count = 0
        for original_row, match, candidate in zip_longest(official,
                rows(work/'matching_results.tsv'),rows(work/'candidate_pairs.tsv')):
            if any(x is None for x in (original_row,match,candidate)):
                raise ValueError('Submission does not cover ALL original test S1 rows (row count differs)')
            sid = original_row['entity_id']
            if sid != match['source1_entity_id'] or sid != candidate['source1_entity_id']:
                raise ValueError(f'Submission S1 ID/order mismatch at original row {count+1}: {sid}')
            matches = [x for x in match['matched_entity_ids'].split(',') if x]
            candidates = [x for x in candidate['candidate_entity_ids'].split(',') if x]
            if len(matches)!=len(set(matches)) or len(candidates)!=len(set(candidates)) or not set(matches)<=set(candidates):
                raise ValueError(f'Duplicate IDs or match outside candidates for {sid}')
            count += 1
    if not count:
        raise ValueError('Original test set is empty')
    result = dict(original=str(Path(original).resolve()), original_test_queries=count,
                  submitted_queries=count, exact_id_and_order_coverage=True)
    dump_json(work/'submission_coverage.json',result)
    print(json.dumps(result,indent=2))
    return result
