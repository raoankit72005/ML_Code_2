"""Offline integration using a tiny real Transformer, LoRA, FAISS and LightGBM."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

class HybridTests(unittest.TestCase):
    def test_real_hybrid_pipeline(self):
        self.run_pipeline(1)

    @unittest.skipUnless(os.environ.get('ER_TEST_DISTRIBUTED')=='1', 'Requires local Gloo networking; set ER_TEST_DISTRIBUTED=1')
    def test_two_process_hybrid_pipeline(self):
        self.run_pipeline(2)

    def run_pipeline(self, workers):
        import torch
        from transformers import BertConfig,BertModel,BertTokenizerFast
        from sentence_transformers import SentenceTransformer,models
        from er_pipeline import encoder
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'bert';base.mkdir()
            vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','name','address','country',':','|','harbor','motors','auto','india','us','road','elm']+[str(i) for i in range(100)]
            (base/'vocab.txt').write_text('\n'.join(vocab))
            tokenizer=BertTokenizerFast(vocab_file=str(base/'vocab.txt'));tokenizer.save_pretrained(base)
            BertModel(BertConfig(vocab_size=len(vocab),hidden_size=24,num_hidden_layers=1,num_attention_heads=4,intermediate_size=32)).save_pretrained(base)
            sentence=SentenceTransformer(modules=[models.Transformer(str(base),max_seq_length=32),models.Pooling(24)])
            saved=root/'sentence';sentence.save(str(saved));del sentence
            raw=root/'original';raw.mkdir()
            fields=['entity_id','business_name','business_address','country']
            for split,count in [('train',60),('test',12)]:
                for source in (1,2,3):
                    with (raw/f'{split}_source{source}.tsv').open('w',newline='') as f:
                        w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader()
                        for i in range(count):w.writerow(dict(entity_id=f'S{source}-{split}-{i}',business_name=f'harbor {"motors" if source!=3 else "auto"} {i}',business_address=f'{i} elm road',country='India' if i%2 else 'US'))
            with (raw/'train_ground_truth.tsv').open('w',newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['source1_entity_id','matched_entity_ids'])
                for i in range(60):w.writerow([f'S1-train-{i}',f'S2-train-{i},S3-train-{i}'])
            cfg=json.loads((ROOT/'configs/hybrid.json').read_text())
            cfg.update(cpu_test_workers=workers,disk_reserve_gb=.01,model_id=str(saved),revision=None,device='cpu',cpu_threads=1,max_length=32,batch_size=8,mini_batch_size=2,
                encode_batch_size=16,log_every=1,checkpoint_every=3,monitor_per_country=3,monitor_references=30,ann_top_k=5,ann_shortlist=10,pq_m=6)
            config=root/'config.json';config.write_text(json.dumps(cfg))
            lgb=json.loads((ROOT/'configs/full_gpu.json').read_text());lgb.update(device_type='cpu',num_threads=1,num_boost_round=8,
                early_stopping_rounds=3,min_data_in_leaf=2,disk_reserve_gb=.01,thresholds=[.5,.75])
            training=root/'training.json';training.write_text(json.dumps(lgb))
            work=root/'work'
            command=[sys.executable,str(ROOT/'hybrid.py'),'--input',str(raw),'--work',str(work),'--config',str(config),'--training-config',str(training)]
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=600)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(json.loads((work/'test/submission_coverage.json').read_text())['submitted_queries'],12)
            self.assertTrue((work/'logs'/('encoder.jsonl' if workers==1 else 'encoder_rank0.jsonl')).exists())
            self.assertTrue((work/'encoder/last.pt').exists())
            import pyarrow.parquet as pq
            features=pq.read_table(work/'train/train_features.parquet',columns=['neural_cosine','block_G']).to_pydict()
            self.assertTrue(all(-1.01<=x<=1.01 for x in features['neural_cosine']))
            self.assertGreater(sum(features['block_G']),0)
            import sqlite3
            conn=sqlite3.connect(work/'clusters.sqlite')
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM (SELECT cluster FROM queries GROUP BY cluster HAVING COUNT(DISTINCT split)>1)').fetchone()[0],0)
            conn.close()
            repeated=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(repeated.returncode,0,repeated.stdout+repeated.stderr)
            self.assertIn('Skip completed encoder',repeated.stdout)
            # Resume training from the stored end-of-epoch checkpoint, without refitting or losing logs.
            (work/'encoder/complete.json').unlink()
            if workers==1:
                encoder.train(work,cfg)
            else:
                resume=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--local-addr=127.0.0.1','--nproc_per_node=2',str(ROOT/'gpu_worker.py'),'encoder','--work',str(work),'--config',str(config)],capture_output=True,text=True,timeout=90)
                self.assertEqual(resume.returncode,0,resume.stdout+resume.stderr)
                self.assertTrue((work/'logs/encoder_rank1.jsonl').exists())
            self.assertTrue((work/'encoder/complete.json').exists())

if __name__=='__main__':unittest.main()
