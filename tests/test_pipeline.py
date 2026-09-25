"""Small deterministic tests; no model is trained and no external data is used."""
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from er_pipeline.common import DEFAULTS, rows, parquet_rows, split_for
from er_pipeline.indexing import build, FIELDS
from er_pipeline.blocking import generate
from er_pipeline.labeling import label
from er_pipeline.features import extract, pair_features
from er_pipeline.tables import export
from er_pipeline.evaluation import f05, evaluate, submit
from er_pipeline.text_features import Tfidf, cosine


def record(eid, name='', address='', country='US'):
    r = dict.fromkeys(FIELDS, '')
    r.update(entity_id=eid, country_key=country, name_basic=name, name_expanded=name,
        name_without_suffix=name, name_latin_folded=name, address_normalized=address,
        address_latin_folded=address, name_missing=int(not name), address_missing=int(not address))
    return r


def write_tsv(path, fields, values):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(values)


class Tests(unittest.TestCase):
    def test_metric_and_missingness(self):
        self.assertEqual(f05(set(), set()), 1)
        self.assertEqual(f05(set(), {'a'}), 0)
        self.assertAlmostEqual(f05({'a','b'}, {'a'}), 5/6)
        self.assertAlmostEqual(f05({'a'}, {'a','x'}), 5/9)
        a, b = record('S1-a', 'abc'), record('S2-a', 'abc', 'road')
        f = pair_features(a, b, 1.0, math.nan)
        self.assertTrue(math.isnan(f['address_levenshtein']))
        self.assertTrue(math.isnan(f['postal_code_exact']))
        self.assertTrue(math.isnan(f['name_address_similarity_product']))
        self.assertEqual(f['address_missing'], 1)
        self.assertEqual(f['name_exact'], 1)
        self.assertEqual(split_for('S1-a','US',True,DEFAULTS), split_for('S1-a','US',True,DEFAULTS))

    def test_tfidf_reordering_unicode(self):
        model = Tfidf(12)
        a = record('S2-a','holloway peak seafood')
        b = record('S3-a','seafood holloway peak')
        model.update(a); model.update(b); model.finish()
        self.assertAlmostEqual(cosine(model.vector(a['name_basic'],'name'), model.vector(b['name_basic'],'name')),1)
        self.assertTrue(model.vector('भारतीय தமிழ்','name'))
        self.assertTrue(math.isnan(cosine({}, {})))

    def test_connected_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data=root/'data'; data.mkdir(); work=root/'work'; work.mkdir()
            cfg=dict(DEFAULTS, block_limit=1, lexical_pool=5, lexical_top_k=5, max_candidates=5, hash_bits=12)
            queries=[record('S1-a','abc motors','105 elm street'),
                     record('S1-empty'),record('S1-missed')]
            refs2=[record('S2-a','abc motors','105 elm street'),record('S2-unseen','unrelated'),
                   record('S2-fr','abc motors','105 elm street','FR')]
            refs3=[record('S3-a','abc motors','105 elm street')]
            for n, values in [(1,queries),(2,refs2),(3,refs3)]:
                write_tsv(data/f'train_source{n}.tsv',FIELDS,values)
            truth=[{'source1_entity_id':'S1-a','matched_entity_ids':'S2-a,S3-a'},
                   {'source1_entity_id':'S1-empty','matched_entity_ids':''},
                   {'source1_entity_id':'S1-missed','matched_entity_ids':'S2-unseen'}]
            gt=data/'train_ground_truth.tsv'
            write_tsv(gt,['source1_entity_id','matched_entity_ids'],truth)
            build(data,work,'train',cfg)
            generate(work,work/'tfidf.npz',cfg)
            pairs=list(parquet_rows(work/'candidate_pairs.parquet'))
            self.assertEqual({p['candidate_entity_id'] for p in pairs}, {'S2-a','S3-a'})
            self.assertEqual(len(pairs),2)
            label(work,gt,cfg)
            recall=json.loads((work/'candidate_recall.json').read_text())['all']
            self.assertAlmostEqual(recall['candidate_recall'],2/3)
            self.assertAlmostEqual(recall['oracle_macro_f05_ceiling'],2/3)
            extract(work,work/'tfidf.npz','train',cfg)
            export(work,'train',cfg)
            exported=list(rows(work/'candidate_pairs.tsv'))
            self.assertEqual(len(exported),3)
            self.assertEqual(exported[1]['candidate_entity_ids'],'')
            self.assertEqual(exported[2]['candidate_entity_ids'],'')
            all_features=list(parquet_rows(work/'pair_features.parquet'))
            self.assertTrue(all(p['label']==1 for p in all_features))
            train={p['source1_entity_id'] for p in parquet_rows(work/'train_features.parquet')}
            val={p['source1_entity_id'] for p in parquet_rows(work/'validation_features.parquet')}
            self.assertFalse(train & val)
            # Force all synthetic queries into validation ONLY to test the metric.
            labels=list(rows(work/'query_labels.tsv.gz'))
            for q in labels:q['dataset_split']='validation'
            import gzip,shutil
            with gzip.open(work/'query_labels.tsv.gz','wt',newline='') as f:
                writer=csv.DictWriter(f,list(labels[0]),delimiter='\t');writer.writeheader();writer.writerows(labels)
            shutil.copyfile(work/'pair_features.parquet',work/'validation_features.parquet')
            probabilities=root/'probabilities.tsv'
            write_tsv(probabilities,['source1_entity_id','candidate_entity_id','match_probability'],
                [dict(source1_entity_id=p['source1_entity_id'],candidate_entity_id=p['candidate_entity_id'],match_probability=0.9) for p in pairs])
            result=evaluate(work,probabilities,[0.5,0.95])
            self.assertAlmostEqual(result['best']['macro_f05'],2/3)
            # Missing score rows must never silently become negative predictions.
            write_tsv(probabilities,['source1_entity_id','candidate_entity_id','match_probability'],[])
            with self.assertRaises(ValueError):evaluate(work,probabilities,[0.5])


if __name__=='__main__':
    unittest.main()
