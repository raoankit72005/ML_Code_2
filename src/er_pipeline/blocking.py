"""Union of bounded blocks A-F, then rank/cap the ACTUAL model candidate set."""
import csv
import json
import math
from collections import Counter
from functools import lru_cache

from .common import connect, ParquetSink, PAIR_SCHEMA, dump_json, open_text
from .text_features import Tfidf, name_text, address_text, cosine, block_keys, encoded


def generate(work, tfidf_path, config, max_queries=None):
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
        query_sql = 'SELECT payload FROM records WHERE source=1 ORDER BY rid'
        cursor = conn.execute(query_sql)
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


def finite(x):
    return x if math.isfinite(x) else 0.0
