"""Union of bounded blocks A-F, then rank/cap the ACTUAL model candidate set."""
import csv
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .common import connect, ParquetSink, PAIR_SCHEMA, dump_json, open_text, parquet_rows
from .text_features import Tfidf, name_text, address_text, cosine, block_keys, encoded


def generate(work, tfidf_path, config, max_queries=None, rid_range=None):
    conn = connect(work / 'index.sqlite', config['sqlite_cache_mb'], readonly=True)
    neural=connect(work/'neural.sqlite',readonly=True) if (work/'neural.sqlite').exists() else None
    tfidf = Tfidf.load(tfidf_path)
    totals = Counter()
    @lru_cache(maxsize=256)
    def reference(rid):
        r = json.loads(conn.execute('SELECT payload FROM records WHERE rid=?', (rid,)).fetchone()[0])
        return r, tfidf.vector(name_text(r), 'name'), tfidf.vector(address_text(r), 'address')
    def search(expression, limit, ranked=False):
        sql = 'SELECT rowid FROM search WHERE search MATCH ?'
        if ranked:
            sql += " AND rank MATCH 'bm25(0.0,0.0,1.0,1.0)' ORDER BY rank"
        return [r[0] for r in conn.execute(sql + ' LIMIT ?', (expression, limit))]
    def field_query(field, tokens):
        return field + ':(' + ' OR '.join(tokens) + ')' if tokens else ''
    qpath = work / 'queries.tsv.gz'
    qpartial = work / 'queries.partial.tsv.gz'
    with open_text(qpartial, 'wt') as qfile, ParquetSink(work / 'candidate_pairs.parquet', PAIR_SCHEMA, config['row_group_size']) as sink:
        writer = csv.DictWriter(qfile, fieldnames=['source1_entity_id', 'country', 'n_candidates',
            'union_candidates', 'overflow_blocks'], delimiter='\t')
        writer.writeheader()
        query_sql = 'SELECT payload FROM records WHERE source=1'
        parameters = ()
        if rid_range is not None:
            query_sql += ' AND rid>=? AND rid<?'
            parameters = rid_range
        cursor = conn.execute(query_sql + ' ORDER BY rid', parameters)
        for index, (payload,) in enumerate(cursor):
            if max_queries is not None and index >= max_queries:
                break
            q = json.loads(payload)
            multiplier = config['india_retrieval_multiplier'] if q['country_key'] == 'IN' else 1
            limit = config['max_candidates'] * multiplier
            qname, qaddress = name_text(q), address_text(q)
            nv, av = tfidf.vector(qname, 'name'), tfidf.vector(qaddress, 'address')
            name_query = field_query('name_terms', tfidf.query_tokens(qname, 'name', config['query_terms']))
            address_query = field_query('address_terms', tfidf.query_tokens(qaddress, 'address', config['query_terms']))
            country_query = field_query('country', [encoded(q['country_key']), encoded('UNKNOWN')]) if q['country_key'] else ''
            def restrict(expression):
                return f'({country_query}) AND ({expression})' if country_query else expression
            candidates, scores = {}, {}
            overflow = []
            def score(rid):
                if rid not in scores:
                    r, rn, ra = reference(rid)
                    scores[rid] = (r, cosine(nv, rn), cosine(av, ra))
                return scores[rid]
            def add(ids, channel):
                for rid in ids:
                    candidates.setdefault(rid, set()).add(channel)
            for channel, tokens in block_keys(q).items():
                if not tokens:
                    continue
                expression = restrict(field_query('blocks', sorted(set(tokens))))
                ids = search(expression, config['block_limit'] + 1)
                if len(ids) > config['block_limit']:
                    overflow.append(channel)
                    # Never take arbitrary first-N members of a huge prefix block.
                    lexical = [s for s in (name_query, address_query) if s]
                    ids = search(expression + ' AND (' + ' OR '.join(lexical) + ')', config['block_limit'], True) if lexical else []
                add(ids, channel)
            # E is approximate nearest-name retrieval: BM25 shortlist -> TF-IDF rerank.
            # F adds address retrieval even when name scripts disagree or city is absent.
            for channel, expression, score_index in [('E', name_query, 1), ('F', address_query, 2)]:
                if not expression:
                    continue
                pool = search(restrict(expression), config['lexical_pool'] * multiplier, True)
                ranked = sorted(pool, key=lambda rid: (-finite(score(rid)[score_index]), score(rid)[0]['entity_id']))
                add(ranked[:config['lexical_top_k'] * multiplier], channel)
            # Whole-word retrieval gets a separate budget so rare typo grams do
            # not consume every search term. Reuses channel E/F provenance.
            if config['word_retrieval']:
                for channel, field, value, score_index in [('E','name_terms',qname,1), ('F','address_terms',qaddress,2)]:
                    tokens = sorted(set(value.split()), key=lambda w: (-len(w),w))[:config['query_terms']]
                    expression = field_query(field, [encoded('w:'+w) for w in tokens])
                    if expression:
                        pool = search(restrict(expression), config['lexical_pool'] * multiplier, True)
                        pool.sort(key=lambda rid: (-finite(score(rid)[score_index]), score(rid)[0]['entity_id']))
                        add(pool[:config['lexical_top_k'] * multiplier], channel)
            neural_scores={}
            if neural:
                for rid,value,nrank in neural.execute('SELECT rid,cosine,rank FROM candidates WHERE sid=?',(q['entity_id'],)):
                    add([rid],'G');neural_scores[rid]=(value,nrank)
            ranked = []
            for rid, channels in candidates.items():
                r, nc, ac = score(rid)
                exact = bool(q['name_basic']) and q['name_basic'] == r['name_basic']
                score_value = .55 * finite(nc) + .30 * finite(ac) + .10 * exact + .01 * len(channels)
                ranked.append((score_value, r['entity_id'], rid, nc, ac, channels))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            # Reserve a small quota for strong name/address evidence independently.
            # Otherwise name-heavy final ranking discards cross-script address matches.
            quota = min(config['diverse_candidates'], limit // 3)
            retained = {}
            for component in (3,4):
                for item in sorted(ranked, key=lambda x: (-finite(x[component]), x[1]))[:quota]:
                    if finite(item[component]) > 0:
                        retained[item[1]] = item
            for item in sorted((x for x in ranked if x[2] in neural_scores),key=lambda x:-neural_scores[x[2]][0])[:min(20,limit//3)]:
                retained[item[1]]=item
            for item in ranked:
                if len(retained) >= limit:
                    break
                retained[item[1]] = item
            selected = sorted(retained.values(), key=lambda x: (-x[0],x[1]))
            for rank, (value, eid, rid, nc, ac, channels) in enumerate(selected, 1):
                r = score(rid)[0]
                row = dict(source1_entity_id=q['entity_id'], candidate_entity_id=eid,
                    candidate_source=r['source'], candidate_rank=rank, retrieval_score=value,
                    name_retrieval_cosine=nc, address_retrieval_cosine=ac)
                row.update({'block_' + ch: int(ch in channels) for ch in 'ABCDEFG'})
                row['neural_cosine'],row['neural_rank']=neural_scores.get(rid,(math.nan,0))
                sink.append(row)
                totals.update({'block_' + ch + '_final_pairs': 1 for ch in channels})
            writer.writerow(dict(source1_entity_id=q['entity_id'], country=q['country_key'],
                n_candidates=len(selected), union_candidates=len(ranked), overflow_blocks=','.join(overflow)))
            totals['queries'] += 1
            totals['pairs'] += len(selected)
            totals['queries_without_candidates'] += not selected
            totals['queries_capped'] += len(ranked) > len(selected)
            totals['queries_with_overflow_blocks'] += bool(overflow)
            if totals['queries'] % config['progress_every'] == 0:
                print(f'blocking: {totals["queries"]:,} queries / {totals["pairs"]:,} pairs', flush=True)
    qpartial.replace(qpath)
    totals['average_candidates'] = totals['pairs'] / max(1, totals['queries'])
    dump_json(work / 'blocking_report.json', dict(totals))
    conn.close()
    if neural:neural.close()
    print(f'Blocking complete: {totals["queries"]:,} queries / {totals["pairs"]:,} pairs', flush=True)


def _chunk(index, start, end, work, tfidf_path, config):
    folder = work / 'blocking_chunks' / f'{index:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    for filename in ('index.sqlite', 'neural.sqlite'):
        source = work / filename
        target = folder / filename
        if source.exists() and not target.exists():
            target.symlink_to(source.resolve())
    report = folder / 'blocking_report.json'
    if not all((folder / name).exists() for name in ('queries.tsv.gz', 'candidate_pairs.parquet', 'blocking_report.json')):
        generate(folder, tfidf_path, config, rid_range=(start, end))
    return index, json.loads(report.read_text())


def generate_parallel(work, tfidf_path, config):
    """Process ordered query ranges; published chunks survive interrupted runs."""
    work = Path(work)
    workers = int(config.get('blocking_workers', 1))
    if workers < 1:
        raise ValueError('blocking_workers must be positive')
    if workers == 1:
        return generate(work, tfidf_path, config)
    chunk_size = int(config.get('blocking_chunk_queries', 10000))
    if chunk_size < 1:
        raise ValueError('blocking_chunk_queries must be positive')
    conn = connect(work / 'index.sqlite', readonly=True)
    lo, hi = conn.execute('SELECT MIN(rid),MAX(rid) FROM records WHERE source=1').fetchone()
    conn.close()
    if lo is None:
        raise ValueError('No source 1 queries')
    ranges = [(i, start, min(start + chunk_size, hi + 1))
              for i, start in enumerate(range(lo, hi + 1, chunk_size))]
    totals = Counter()
    reports = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_chunk, i, start, end, work, tfidf_path, config)
                   for i, start, end in ranges]
        for future in as_completed(futures):
            i, report = future.result()
            reports[i] = report
            print(f'blocking: completed {len(reports)}/{len(ranges)} chunks; '
                  f'{sum(r["queries"] for r in reports.values()):,} queries', flush=True)
    qpath = work / 'queries.partial.tsv.gz'
    with open_text(qpath, 'wt') as output, ParquetSink(work / 'candidate_pairs.parquet', PAIR_SCHEMA, config['row_group_size']) as sink:
        for i, _, _ in ranges:
            folder = work / 'blocking_chunks' / f'{i:06d}'
            with open_text(folder / 'queries.tsv.gz') as source:
                header = next(source)
                if i == 0:
                    output.write(header)
                for line in source:
                    output.write(line)
            for row in parquet_rows(folder / 'candidate_pairs.parquet'):
                sink.append(row)
            totals.update({key: value for key, value in reports[i].items()
                           if key != 'average_candidates'})
    qpath.replace(work / 'queries.tsv.gz')
    totals['average_candidates'] = totals['pairs'] / max(1, totals['queries'])
    dump_json(work / 'blocking_report.json', dict(totals))
    print(f'Blocking complete: {totals["queries"]:,} queries / {totals["pairs"]:,} pairs', flush=True)


def finite(x):
    return x if math.isfinite(x) else 0.0
