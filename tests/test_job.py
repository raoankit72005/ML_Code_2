import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'sagemaker'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class JobTests(unittest.TestCase):
    def test_request_has_full_data_and_four_gpu_instance(self):
        m=module('launch')
        args=argparse.Namespace(role='arn:aws:iam::123456789012:role/Example',input='s3://bucket/data.zip',output='s3://bucket/out',name='test-job',image='example/image:tag',volume_gb=1000,max_hours=24)
        r=m.request(args)
        self.assertEqual(r['ResourceConfig']['InstanceType'],'ml.g6e.12xlarge')
        self.assertEqual(r['ResourceConfig']['InstanceCount'],1)
        self.assertEqual(r['CheckpointConfig']['LocalPath'],'/opt/ml/checkpoints')
        self.assertFalse(r['EnableManagedSpotTraining'])
        self.assertIn('test-job',r['OutputDataConfig']['S3OutputPath'])
        args.profile='g5';args.image='example/image:sm86-test';args.volume_gb=None
        small=m.request(args)
        self.assertEqual(small['ResourceConfig']['InstanceType'],'ml.g5.xlarge')
        self.assertEqual(small['Environment']['ER_PROFILE'],'g5')
        self.assertEqual(m.request(args)['ResourceConfig']['VolumeSizeInGB'],200)
        args.volume_gb=1000
        with self.assertRaises(ValueError):m.request(args)
        args.volume_gb=200
        args.image='example/image:sm89-test'
        with self.assertRaises(ValueError):m.request(args)
        args.image='example/image:sm86-test'
        args.name='bad name'
        with self.assertRaises(ValueError):m.request(args)

    def test_channel_ambiguity_rejected(self):
        m=module('train')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'data.zip').touch()
            self.assertEqual(m.discover(p),p/'data.zip')
            (p/'train_source1.tsv').touch()
            with self.assertRaises(ValueError):m.discover(p)

    def test_export_excludes_training_arrays(self):
        m=module('train')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);work=root/'work'
            for folder in ('model/data_cache','encoder','train','test','logs'):
                (work/folder).mkdir(parents=True,exist_ok=True)
            (work/'model/data_cache/train.features.f32').write_bytes(b'large training cache')
            (work/'model/lightgbm_model.txt').write_text('model')
            (work/'train/tfidf.npz').write_bytes(b'tfidf')
            (work/'test/matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\n')
            m.copy_outputs(work,root/'model-output',root/'data-output')
            self.assertFalse((root/'model-output/matcher/data_cache').exists())
            self.assertTrue((root/'model-output/matcher/lightgbm_model.txt').exists())
            self.assertTrue((root/'data-output/matching_results.tsv').exists())

    def test_disjoint_ranges_and_training_tail(self):
        import sys
        sys.path.insert(0,str(ROOT/'src'))
        from er_pipeline.parallel_embed import bounds
        from er_pipeline.distributed import rank_batch
        for n in (0,1,3,4,5,33):
            joined=[i for r in range(4) for i in range(*bounds(n,r,4))]
            self.assertEqual(joined,list(range(n)))
        b=list(range(17))
        self.assertEqual(sorted(x for r in range(4) for x in rank_batch(b,r,4)),b)
        self.assertEqual(rank_batch([1,2],3,4),[1,2])

if __name__=='__main__':unittest.main()
