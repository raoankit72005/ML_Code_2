"""Exercise actual LightGBM training, disk reload, prediction and score coverage."""
import csv
import gzip
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))
from er_pipeline import modeling, evaluation
from er_pipeline.common import dump_json, parquet_rows, rows


class ModelingTests(unittest.TestCase):
    def fixture(self, root):
        train=root/'train'; train.mkdir(); test=root/'test'; test.mkdir()
        names=['name_exact','address_levenshtein']
        schema=pa.schema([('source1_entity_id',pa.string()),('candidate_entity_id',pa.string()),
            ('name_exact',pa.float32()),('address_levenshtein',pa.float32()),
            ('label',pa.int8()),('dataset_split',pa.string())])
        qlabels=[]
        for group, count in [('train',60),('validation',12)]:
            pairs=[]
            for i in range(count):
                sid=f'S1-{group}-{i}'
                true=f'S2-{group}-{i}'
                for match in (0,1):
                    pairs.append(dict(source1_entity_id=sid,candidate_entity_id=true if match else f'S3-{group}-{i}',
                        name_exact=float(match),address_levenshtein=math.nan if i%3==0 else (0.9 if match else 0.1),
                        label=match,dataset_split=group))
                if group=='validation':
                    qlabels.append(dict(source1_entity_id=sid,country='US',dataset_split=group,matched_entity_ids=true))
            pq.write_table(pa.Table.from_pylist(pairs,schema=schema),train/f'{group}_features.parquet')
        # Zero-candidate singleton and missed-positive query must remain in metric denominator.
        qlabels += [dict(source1_entity_id='S1-singleton',country='US',dataset_split='validation',matched_entity_ids=''),
                    dict(source1_entity_id='S1-missed',country='US',dataset_split='validation',matched_entity_ids='S2-missed')]
        with gzip.open(train/'query_labels.tsv.gz','wt',newline='') as f:
            w=csv.DictWriter(f,list(qlabels[0]),delimiter='\t');w.writeheader();w.writerows(qlabels)
        for folder in (train,test):dump_json(folder/'feature_columns.json',{'features':names})
        np.savez(train/'tfidf.npz',fixture=np.array([1,2]))
        test_schema=pa.schema([f for f in schema if f.name not in ('label','dataset_split')])
        test_rows=[dict(source1_entity_id='S1-test',candidate_entity_id='S2-test',name_exact=1.,address_levenshtein=math.nan),
                   dict(source1_entity_id='S1-test',candidate_entity_id='S3-test',name_exact=0.,address_levenshtein=.1)]
        pq.write_table(pa.Table.from_pylist(test_rows,schema=test_schema),test/'test_features.parquet')
        with gzip.open(test/'queries.tsv.gz','wt',newline='') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['source1_entity_id','n_candidates']);w.writerow(['S1-test',2]);w.writerow(['S1-empty',0])
        return names

    def test_fit_predict_threshold_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);names=self.fixture(root)
            cfg=modeling.load_config(overrides=dict(num_threads=1,num_boost_round=12,early_stopping_rounds=3,
                min_data_in_leaf=2,max_train_pairs=80,max_early_stopping_pairs=10,batch_size=7,thresholds=[.3,.5,.7]))
            x,y,info=modeling.load_training_rows(root/'train/train_features.parquet',names,80,42,'train',7)
            x2,y2,_=modeling.load_training_rows(root/'train/train_features.parquet',names,80,42,'train',13)
            np.testing.assert_equal(x,x2);np.testing.assert_equal(y,y2)
            self.assertEqual(info['selected_pairs'],80)
            self.assertTrue(np.isnan(x).any())
            metadata=modeling.train(root,cfg)
            self.assertEqual(metadata['train_rows']['selected_pairs'],80)
            self.assertEqual(metadata['early_stopping_rows']['selected_pairs'],10)
            self.assertEqual(metadata['final_validation_pairs'],24)
            self.assertEqual(metadata['validation_query_counts']['all'],14)
            # The unretrievable positive prevents a perfect macro score.
            self.assertLessEqual(metadata['validation_macro_f05'],13/14+1e-12)
            self.assertEqual(modeling.train(root,cfg)['model_sha256'],metadata['model_sha256'])
            output,loaded=modeling.predict(root,'test')
            predictions=list(parquet_rows(output))
            self.assertEqual(len(predictions),2)
            self.assertTrue(all(0<=p['match_probability']<=1 for p in predictions))
            evaluation.submit(root/'test',output,loaded['decision_threshold'])
            results=list(rows(root/'test/matching_results.tsv'))
            self.assertEqual(len(results),2)
            self.assertEqual(results[1]['matched_entity_ids'],'')
            self.assertTrue(set(filter(None,results[0]['matched_entity_ids'].split(',')))<={'S2-test','S3-test'})
            # Valid empty candidate tables still yield a header-only probability artifact.
            pq.write_table(pa.Table.from_pylist([],schema=pq.read_schema(root/'test/test_features.parquet')),root/'test/test_features.parquet')
            empty,_=modeling.predict(root,'test')
            self.assertEqual(pq.ParquetFile(empty).metadata.num_rows,0)
            # Prevent mixing a model and a differently ordered feature schema.
            dump_json(root/'test/feature_columns.json',{'features':names[::-1]})
            with self.assertRaises(ValueError):modeling.predict(root,'test')
            with self.assertRaises(ValueError):modeling.train(root,dict(cfg,num_boost_round=13))

    def test_forbidden_features_and_wrong_split(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);names=self.fixture(root)
            with self.assertRaises(ValueError):
                modeling.load_training_rows(root/'train/train_features.parquet',names,80,42,'validation',10)
            dump_json(root/'train/feature_columns.json',{'features':['label','name_exact']})
            with self.assertRaises(ValueError):modeling.feature_names(root/'train')
        with self.assertRaises(ValueError):modeling.load_config(overrides={'max_train_pairs':0})


if __name__=='__main__':
    unittest.main()
