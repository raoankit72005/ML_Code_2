"""Batched disk embeddings and bounded country/source IVF-PQ retrieval."""
import json
import math
import os
from pathlib import Path

import numpy as np

from .common import connect,dump_json
from .resources import available_ram,check_disk
from .tracking import Tracker


def embed(work,root,config):
    from . import encoder
    from .modeling import file_identity,sha256
    manifest=json.loads((root/'encoder/complete.json').read_text())
    signature=dict(index=file_identity(work/'index.sqlite'),encoder=manifest,config=config)
    directory=work/'embeddings';directory.mkdir(exist_ok=True)
    marker=directory/'manifest.json';path=directory/'vectors.f16'
    conn=connect(work/'index.sqlite',readonly=True)
    count,maximum=conn.execute('SELECT COUNT(*),MAX(rid) FROM records').fetchone()
    if count!=maximum:raise ValueError('Embedding lookup requires contiguous record IDs')
    saved=json.loads(marker.read_text()) if marker.exists() else None
    if saved and saved['signature']!=signature:raise ValueError('Embedding inputs changed; use new work folder')
    if saved and saved.get('complete'):
        if path.stat().st_size!=count*saved['dimension']*2:raise ValueError('Embedding file size mismatch')
        conn.close();return
    model=encoder.load(config,root/'encoder/best.pt' if manifest['mode']=='lora' else None,manifest['mode']=='lora')
    dim=model.get_sentence_embedding_dimension();completed=saved['rows'] if saved else 0
    check_disk(directory,max(0,count*dim*2-(path.stat().st_size if path.exists() else 0)),5)
    if path.exists() and not saved:path.unlink()
    if saved and (not path.exists() or path.stat().st_size<completed*dim*2):raise ValueError('Partial embedding cache was truncated')
    tracker=Tracker(root,'embeddings_'+work.name)
    cursor=conn.execute('SELECT rid,payload FROM records WHERE rid>? ORDER BY rid',(completed,))
    with path.open('r+b' if path.exists() else 'wb') as f:
        f.truncate(completed*dim*2);f.seek(completed*dim*2)
        while True:
            records=cursor.fetchmany(config['encode_batch_size'])
            if not records:break
            batch_size=config['encode_batch_size']
            while True:
                try:
                    vectors=model.encode([encoder.text(json.loads(r[1])) for r in records],
                        batch_size=batch_size,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
                    break
                except __import__('torch').cuda.OutOfMemoryError:
                    __import__('torch').cuda.empty_cache()
                    if batch_size<=1:raise
                    batch_size=max(1,batch_size//2)
                    tracker.log(completed,oom_retry=1,inference_batch_size=batch_size)
            if not np.isfinite(vectors).all():raise ValueError('Non-finite embeddings')
            check_disk(directory,vectors.nbytes,5)
            vectors.astype(np.float16).tofile(f);f.flush();os.fsync(f.fileno())
            completed+=len(records)
            dump_json(marker,dict(signature=signature,rows=completed,dimension=dim,complete=False))
            if completed%(config['encode_batch_size']*100)==0:tracker.log(completed,records=completed,total_records=count)
    dump_json(marker,dict(signature=signature,rows=count,dimension=dim,complete=True))
    tracker.log(completed,records=completed,complete=1);tracker.close();conn.close()


def vectors(work):
    info=json.loads((work/'embeddings/manifest.json').read_text())
    if not info['complete']:raise ValueError('Embeddings are incomplete')
    return np.memmap(work/'embeddings/vectors.f16',mode='r',dtype=np.float16,shape=(info['rows'],info['dimension']))


def unit(x):
    x=np.array(x,dtype=np.float32,copy=True)
    norms=np.linalg.norm(x,axis=-1,keepdims=True)
    return x/np.maximum(norms,1e-12)


def retrieve(work,root,config):
    import faiss
    faiss.omp_set_num_threads(config['cpu_threads'])
    vec=vectors(work);dim=vec.shape[1]
    refs=connect(work/'index.sqlite',readonly=True)
    output=work/'neural.sqlite.partial'
    if output.exists():output.unlink()
    db=connect(output,64)
    db.execute('CREATE TABLE candidates(sid TEXT,rid INTEGER,cosine REAL,rank INTEGER,PRIMARY KEY(sid,rid)) WITHOUT ROWID')
    tracker=Tracker(root,'retrieval_'+work.name)
    partitions=list(refs.execute("SELECT DISTINCT json_extract(payload,'$.country_key'),source FROM records WHERE source IN (2,3)"))
    for part,(country,source) in enumerate(partitions,1):
        clause="source=? AND json_extract(payload,'$.country_key')=?"
        count=refs.execute('SELECT COUNT(*) FROM records WHERE '+clause,(source,country)).fetchone()[0]
        rng=np.random.default_rng(config['seed'])
        # Reservoir reference IDs, not all text/vectors, for PQ training.
        sample=[]
        for i,(rid,) in enumerate(refs.execute('SELECT rid FROM records WHERE '+clause,(source,country))):
            if len(sample)<config['ann_training_rows']:sample.append(rid)
            else:
                j=int(rng.integers(i+1))
                if j<len(sample):sample[j]=rid
        train_vectors=unit(vec[np.array(sample)-1])
        nlist=min(config['nlist'],max(1,len(sample)//40))
        small=count<10000
        estimated=(count*(dim*4+8) if small else count*(config['pq_m']+8)*1.5)+train_vectors.nbytes*3+256*1024**2
        if estimated>available_ram()*.5:raise RuntimeError('ANN partition exceeds RAM budget; use smaller shards or more RAM')
        if small:index=faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        else:
            if dim%config['pq_m']:raise ValueError('pq_m must divide embedding dimension')
            index=faiss.IndexIVFPQ(faiss.IndexFlatIP(dim),dim,nlist,config['pq_m'],8,faiss.METRIC_INNER_PRODUCT)
            index.train(train_vectors);index.nprobe=min(config['nprobe'],nlist)
        del train_vectors
        cursor=refs.execute('SELECT rid FROM records WHERE '+clause+' ORDER BY rid',(source,country))
        while batch:=cursor.fetchmany(4096):
            ids=np.array([r[0] for r in batch],dtype=np.int64)
            index.add_with_ids(unit(vec[ids-1]),ids)
        # Known queries search their country plus unknown references; unknown queries search all partitions.
        query="SELECT rid,entity_id FROM records WHERE source=1"
        params=()
        if country:
            query+=" AND json_extract(payload,'$.country_key') IN (?, '')";params=(country,)
        cursor=refs.execute(query+' ORDER BY rid',params)
        queried=0
        while batch:=cursor.fetchmany(64):
            qids=np.array([r[0] for r in batch],dtype=np.int64)
            qvec=unit(vec[qids-1]);_,neighbors=index.search(qvec,min(count,config['ann_shortlist']))
            for (_,sid),q,ids in zip(batch,qvec,neighbors):
                ids=ids[ids!=-1];scores=unit(vec[ids-1])@q
                order=np.argsort(-scores,kind='stable')[:config['ann_top_k']]
                db.executemany('INSERT INTO candidates VALUES(?,?,?,?)',[(sid,int(ids[j]),float(scores[j]),rank) for rank,j in enumerate(order,1)])
            queried+=len(batch);db.commit();check_disk(work)
            if queried%6400==0:tracker.log(part*10**9+queried,partition=part,queries_in_partition=queried)
        del index
        tracker.log(part*10**9+queried,partition=part,partitions=len(partitions),queries_in_partition=queried)
    db.close();refs.close();vec._mmap.close();output.replace(work/'neural.sqlite');tracker.close()
