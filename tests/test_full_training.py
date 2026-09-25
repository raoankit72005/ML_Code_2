"""All-pair native training, resource failures, and original-test coverage."""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from er_pipeline import modeling, resources, coverage, evaluation
from er_pipeline.disk_training import stage_all
import test_modeling


class FullTrainingTests(unittest.TestCase):
    def test_all_pairs_native_lightgbm_and_cached_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            names=test_modeling.ModelingTests().fixture(root)
            cfg=modeling.load_config(overrides=dict(full_data=True,device_type='cpu',num_threads=1,
                num_boost_round=8,early_stopping_rounds=3,min_data_in_leaf=2,
                max_train_pairs=1,max_early_stopping_pairs=1,batch_size=7,disk_reserve_gb=.01,
                thresholds=[.3,.5,.7]))
            metadata=modeling.train(root,cfg)
            self.assertEqual(metadata['train_rows']['selected_pairs'],120)
            self.assertEqual(metadata['early_stopping_rows']['selected_pairs'],24)
            self.assertFalse(metadata['train_rows']['sampled'])
            seq,y,stats=stage_all(root/'train/train_features.parquet',names,'train',
                root/'model/data_cache',3,.01)
            x,labels,_=modeling.load_training_rows(root/'train/train_features.parquet',names,120,42,'train',9)
            np.testing.assert_equal(seq[:],x)
            np.testing.assert_equal(y,labels)
            self.assertEqual(seq[0].dtype,np.float64)
            seq.close(); y._mmap.close()

    def test_no_silent_sample_or_gpu_fallback_on_resource_shortage(self):
        cfg=modeling.load_config(overrides=dict(full_data=True,device_type='cuda'))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(resources,'available_ram',return_value=128*1024**3), \
                 patch.object(resources,'gpu_memory',return_value=dict(name='test',free_bytes=1024**3,total_bytes=1024**3)):
                with self.assertRaisesRegex(RuntimeError,'GPU estimate'):
                    resources.training_plan(10_000_000,2_000_000,47,cfg,Path(directory))
            with patch.object(resources,'available_ram',return_value=32*1024**2):
                cpu=dict(cfg,device_type='cpu')
                with self.assertRaisesRegex(RuntimeError,'RAM estimate'):
                    resources.training_plan(10_000_000,2_000_000,47,cpu,Path(directory))

    def test_original_test_coverage_and_prediction_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            test_modeling.ModelingTests().fixture(root)
            raw=root/'raw';raw.mkdir()
            (raw/'test_source1.tsv').write_text('entity_id\nS1-test\nS1-empty\n')
            work=root/'test'
            (work/'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\nS1-test\tS2-test\nS1-empty\t\n')
            (work/'candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-test\tS2-test,S3-test\nS1-empty\t\n')
            self.assertEqual(coverage.validate(root,raw)['submitted_queries'],2)
            (raw/'test_source1.tsv').write_text('entity_id\nS1-test\nS1-empty\nS1-missing\n')
            with self.assertRaisesRegex(ValueError,'ALL original'):
                coverage.validate(root,raw)
            pairs=[dict(source1_entity_id='S1-test',candidate_entity_id=e,match_probability=.8) for e in ['S3-test','S2-test']]
            path=work/'bad.parquet';pq.write_table(pa.Table.from_pylist(pairs),path)
            with self.assertRaisesRegex(ValueError,'order/IDs'):
                evaluation.submit(work,path,.5)
            # Failed formatting must not replace the last completed file.
            self.assertIn('S2-test',(work/'matching_results.tsv').read_text())
            (raw/'SAMPLE_README.txt').write_text('sample')
            with self.assertRaisesRegex(ValueError,'sample'):
                coverage.validate(root,raw)

    def test_full_command_clean_train_predict_and_audit(self):
        import subprocess
        project=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); raw=root/'original'; raw.mkdir()
            fields=['entity_id','business_name','business_address','country']
            for split,count in [('train',60),('test',12)]:
                for source in (1,2,3):
                    with (raw/f'{split}_source{source}.tsv').open('w',newline='') as f:
                        writer=csv.DictWriter(f,fields,delimiter='\t');writer.writeheader()
                        for i in range(count):
                            writer.writerow(dict(entity_id=f'S{source}-{split}-{i}',
                                business_name=f'harbor motors branch {i}' if source!=3 else f'harbor auto spare {i}',
                                business_address=f'{i+1} elm road city',country='India' if i%2 else 'US'))
            with (raw/'train_ground_truth.tsv').open('w',newline='') as f:
                writer=csv.writer(f,delimiter='\t');writer.writerow(['source1_entity_id','matched_entity_ids'])
                for i in range(60):writer.writerow([f'S1-train-{i}',f'S2-train-{i}'])
            cfg=modeling.load_config(overrides=dict(full_data=True,device_type='cpu',num_threads=1,
                num_boost_round=8,early_stopping_rounds=3,min_data_in_leaf=2,batch_size=13,
                max_train_pairs=1,max_early_stopping_pairs=1,disk_reserve_gb=.01,thresholds=[.5,.75]))
            config=root/'cpu.json';config.write_text(json.dumps(cfg))
            work=root/'work'
            command=[sys.executable,str(project/'run.py'),'full','--input',str(raw),'--work',str(work),'--training-config',str(config)]
            result=subprocess.run(command,cwd=project,text=True,capture_output=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            metadata=json.loads((work/'model/model_metadata.json').read_text())
            self.assertFalse(metadata['train_rows']['sampled'])
            self.assertEqual(metadata['train_rows']['selected_pairs'],pq.ParquetFile(work/'train/train_features.parquet').metadata.num_rows)
            self.assertEqual(json.loads((work/'test/submission_coverage.json').read_text())['submitted_queries'],12)
            repeat=subprocess.run(command,cwd=project,text=True,capture_output=True,timeout=120)
            self.assertEqual(repeat.returncode,0,repeat.stdout+repeat.stderr)
            self.assertIn('Skip completed training',repeat.stdout)

    @unittest.skipUnless(os.environ.get('RUN_CUDA_TESTS')=='1','Requires a CUDA-enabled LightGBM build and NVIDIA GPU')
    def test_real_cuda_backend(self):
        resources.preflight(modeling.load_config(overrides=dict(device_type='cuda')))


if __name__=='__main__':
    unittest.main()
