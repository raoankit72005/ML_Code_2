"""Exercise the compressed IVF-PQ branch, not just tiny exact indexes."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from er_pipeline import neural
from er_pipeline.common import connect, dump_json

class ANNTests(unittest.TestCase):
    def test_ivfpq_rerank(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);work=root/'train';work.mkdir();(work/'embeddings').mkdir()
            rng=np.random.default_rng(13)
            refs=neural.unit(rng.normal(size=(10000,8)))
            vectors=np.vstack([refs[17],refs]).astype(np.float16)
            vectors.tofile(work/'embeddings/vectors.f16')
            dump_json(work/'embeddings/manifest.json',dict(rows=10001,dimension=8,complete=True))
            conn=connect(work/'index.sqlite')
            conn.execute('CREATE TABLE records(rid INTEGER PRIMARY KEY,entity_id TEXT,source INTEGER,payload TEXT)')
            conn.executemany('INSERT INTO records VALUES(?,?,?,?)',
                ((i+1,'query' if i==0 else f'ref-{i}',1 if i==0 else 2,json.dumps({'country_key':'US'})) for i in range(10001)))
            conn.commit();conn.close()
            neural.retrieve(work,root,dict(cpu_threads=1,seed=13,ann_training_rows=10000,nlist=8,nprobe=8,pq_m=2,ann_shortlist=128,ann_top_k=5))
            conn=connect(work/'neural.sqlite',readonly=True)
            result=conn.execute('SELECT rid,cosine,rank FROM candidates ORDER BY rank').fetchall()
            self.assertEqual(len(result),5)
            self.assertEqual(result[0][0],19)
            self.assertAlmostEqual(result[0][1],1.,places=5)
            conn.close()
