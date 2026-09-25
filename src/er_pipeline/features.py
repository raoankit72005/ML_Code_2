"""Pairwise features. Similarities are 0..1; missing evidence is IEEE NaN."""
import math
from functools import lru_cache

import pyarrow as pa
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from .common import connect, lookup, parquet_rows, ParquetSink, PAIR_SCHEMA, LABEL_SCHEMA, dump_json
from .text_features import grams

FEATURES = [
    'name_token_sort_ratio', 'address_token_sort_ratio',
    'name_levenshtein_ratio', 'name_jaro_winkler', 'name_token_jaccard',
    'name_tfidf_cosine', 'name_char_ngram_similarity', 'name_exact',
    'name_without_suffix_similarity', 'name_expanded_similarity',
    'address_levenshtein', 'address_edit_distance', 'address_token_jaccard',
    'address_tfidf_cosine', 'address_char_similarity', 'postal_code_exact',
    'city_exact', 'state_exact', 'house_number_exact', 'country_exact',
    'address_exact', 'name_address_similarity_product', 'name_missing',
    'address_missing', 'name_missing_left', 'name_missing_right',
    'address_missing_left', 'address_missing_right', 'name_missing_both',
    'address_missing_both', 'postal_missing', 'city_missing', 'state_missing',
    'house_number_missing', 'name_length_ratio', 'address_length_ratio',
]


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else math.nan


def exact(a, b):
    return float(a == b) if a and b else math.nan


def lev(a, b):
    return Levenshtein.normalized_similarity(a, b) if a and b else math.nan


def length_ratio(a, b):
    return min(len(a), len(b)) / max(len(a), len(b)) if a and b else math.nan


def pair_features(a, b, name_cosine, address_cosine):
    # Text is already normalized. No ASCII stripping or pair-dependent fitting.
    an, bn = a['name_basic'], b['name_basic']
    aa, ba = a['address_normalized'], b['address_normalized']
    name_sim, addr_sim = lev(an, bn), lev(aa, ba)
    f = dict(
        name_token_sort_ratio=fuzz.token_sort_ratio(an,bn)/100 if an and bn else math.nan,
        address_token_sort_ratio=fuzz.token_sort_ratio(aa,ba)/100 if aa and ba else math.nan,
        name_levenshtein_ratio=name_sim,
        name_jaro_winkler=JaroWinkler.normalized_similarity(an, bn) if an and bn else math.nan,
        name_token_jaccard=jaccard(set(an.split()), set(bn.split())),
        name_tfidf_cosine=name_cosine,
        name_char_ngram_similarity=jaccard(grams(an), grams(bn)),
        name_exact=exact(an, bn),
        name_without_suffix_similarity=lev(a['name_without_suffix'], b['name_without_suffix']),
        name_expanded_similarity=lev(a['name_expanded'], b['name_expanded']),
        address_levenshtein=addr_sim,
        address_edit_distance=Levenshtein.distance(aa, ba) if aa and ba else math.nan,
        address_token_jaccard=jaccard(set(aa.split()), set(ba.split())),
        address_tfidf_cosine=address_cosine,
        address_char_similarity=jaccard(grams(aa), grams(ba)),
        country_exact=exact(a['country_key'], b['country_key']),
        address_exact=exact(aa, ba),
        name_address_similarity_product=name_sim * addr_sim,
        name_length_ratio=length_ratio(an, bn), address_length_ratio=length_ratio(aa, ba),
    )
    for stem, column in [('postal', 'postal_code_candidate'), ('city', 'city_candidate'),
                         ('state', 'state_candidate'), ('house_number', 'house_number_zero_normalized')]:
        f[('postal_code' if stem == 'postal' else stem) + '_exact'] = exact(a[column], b[column])
        f[stem + '_missing'] = int(not a[column] or not b[column])
    for field in ('name', 'address'):
        left, right = int(a[field + '_missing']), int(b[field + '_missing'])
        f[field + '_missing'] = int(bool(left or right))
        f[field + '_missing_left'] = left
        f[field + '_missing_right'] = right
        f[field + '_missing_both'] = int(bool(left and right))
    return f


def extract(work, tfidf_path, split, config):
    vec=None
    if (work/'embeddings/manifest.json').exists():
        from .neural import vectors,unit
        vec=vectors(work)
    conn = connect(work / 'index.sqlite', config['sqlite_cache_mb'], readonly=True)
    @lru_cache(maxsize=256)
    def reference(eid):
        r = lookup(conn, eid)
        return r
    schema = pa.schema(list(LABEL_SCHEMA if split == 'train' else PAIR_SCHEMA) +
                       [pa.field(k, pa.float32()) for k in FEATURES])
    source = work / ('labeled_pairs.parquet' if split == 'train' else 'candidate_pairs.parquet')
    current, q, count = None, None, 0
    with ParquetSink(work / 'pair_features.parquet', schema, config['row_group_size']) as sink:
        for pair in parquet_rows(source):
            sid = pair['source1_entity_id']
            if sid != current:
                q = lookup(conn, sid)
                current = sid
                if vec is not None:
                    qr=conn.execute('SELECT rid FROM records WHERE entity_id=?',(sid,)).fetchone()[0]
                    query_vector=unit(vec[qr-1])
            r = reference(pair['candidate_entity_id'])
            if vec is not None:
                rr=conn.execute('SELECT rid FROM records WHERE entity_id=?',(pair['candidate_entity_id'],)).fetchone()[0]
                pair['neural_cosine']=float(query_vector@unit(vec[rr-1]))
            f = pair_features(q, r, pair['name_retrieval_cosine'], pair['address_retrieval_cosine'])
            sink.append(dict(pair, **f))
            count += 1
            if count % (config['progress_every'] * 10) == 0:
                print(f'features: {count:,} pairs', flush=True)
    conn.close()
    if vec is not None:vec._mmap.close()
    model_columns = FEATURES + ['candidate_source', 'candidate_rank', 'retrieval_score',
        'name_retrieval_cosine', 'address_retrieval_cosine'] + ['block_' + c for c in 'ABCDEFG']
    model_columns+=['neural_cosine','neural_rank']
    dump_json(work / 'feature_columns.json', dict(features=model_columns, target='label',
        never_features=['source1_entity_id', 'candidate_entity_id', 'dataset_split'],
        missing_value='NaN; do not fill with zero', pair_count=count))
    print(f'Features complete: {count:,} pairs', flush=True)
