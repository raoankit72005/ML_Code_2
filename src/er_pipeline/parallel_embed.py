"""Encode disjoint contiguous record ranges; merge bounded shards in record-ID order."""
import json
import os
import shutil
from pathlib import Path
import numpy as np
from .common import connect,dump_json
from .resources import check_disk
from .modeling import file_identity


def bounds(count, rank, world):
    return count*rank//world, count*(rank+1)//world


def worker(work,root,config,rank,world):
    import torch
    from . import encoder
    from .tracking import Tracker
    if config['device']=='cuda': torch.cuda.set_device(rank)
    torch.set_num_threads(max(1,config['cpu_threads']//world))
    conn=connect(work/'index.sqlite',readonly=True)
    count,maximum=conn.execute('SELECT COUNT(*),MAX(rid) FROM records').fetchone()
    if count!=maximum:raise ValueError('Noncontiguous record IDs')
    begin,end=bounds(count,rank,world)
    selected=json.loads((root/'encoder/complete.json').read_text())
    signature=dict(index=file_identity(work/'index.sqlite'),encoder=selected,config=config,world=world)
    directory=work/'embeddings';directory.mkdir(exist_ok=True)
    marker=directory/f'rank{rank}.json';path=directory/f'rank{rank}.f16'
    saved=json.loads(marker.read_text()) if marker.exists() else None
    if saved and saved['signature']!=signature:raise ValueError('Embedding shard signature mismatch')
    if saved and saved['complete']:
        if path.stat().st_size!=(end-begin)*saved['dimension']*2:raise ValueError('Invalid complete shard')
        conn.close();return
    model=encoder.load(config,root/'encoder/best.pt' if selected['mode']=='lora' else None,selected['mode']=='lora')
    dim=model.get_sentence_embedding_dimension();completed=saved['rows'] if saved else 0
    if completed and (not path.exists() or path.stat().st_size<completed*dim*2):raise ValueError('Truncated embedding shard')
    tracker=Tracker(root,f'embeddings_{work.name}_rank{rank}')
    cursor=conn.execute('SELECT payload FROM records WHERE rid>? AND rid<=? ORDER BY rid',(begin+completed,end))
    size=config['encode_batch_size']
    with path.open('r+b' if path.exists() else 'wb') as f:
        f.truncate(completed*dim*2);f.seek(completed*dim*2)
        while True:
            records=cursor.fetchmany(size)
            if not records:break
            texts=[encoder.text(json.loads(r[0])) for r in records]
            while True:
                try:
                    out=model.encode(texts,batch_size=size,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if size<=1:raise
                    size=max(1,size//2);tracker.log(completed,oom_retry=1,batch_size=size)
            if not np.isfinite(out).all():raise ValueError('Non-finite embeddings')
            check_disk(directory,out.nbytes,config.get('disk_reserve_gb',10))
            out.astype(np.float16).tofile(f);f.flush();os.fsync(f.fileno());completed+=len(records)
            dump_json(marker,dict(signature=signature,rows=completed,dimension=dim,complete=completed==end-begin))
            if completed%max(1,size*20)==0:tracker.log(completed,records=completed,total=end-begin)
    dump_json(marker,dict(signature=signature,rows=completed,dimension=dim,complete=True))
    tracker.log(completed,records=completed,total=end-begin,complete=1);tracker.close();conn.close()


def merge(work,world,config):
    directory=work/'embeddings';meta=[json.loads((directory/f'rank{r}.json').read_text()) for r in range(world)]
    if not all(m['complete'] and m['signature']==meta[0]['signature'] and m['dimension']==meta[0]['dimension'] for m in meta):
        raise ValueError('Embedding shards inconsistent')
    total=sum(m['rows'] for m in meta);dim=meta[0]['dimension']
    check_disk(directory,total*dim*2,config.get('disk_reserve_gb',10))
    temp=directory/'vectors.partial'
    with temp.open('wb') as out:
        for rank,m in enumerate(meta):
            path=directory/f'rank{rank}.f16'
            if path.stat().st_size!=m['rows']*dim*2:raise ValueError('Shard size mismatch')
            with path.open('rb') as src:shutil.copyfileobj(src,out,16*1024**2)
        out.flush();os.fsync(out.fileno())
    temp.replace(directory/'vectors.f16')
    dump_json(directory/'manifest.json',dict(signature=meta[0]['signature'],rows=total,dimension=dim,complete=True))
    # Merge temporarily doubles embedding storage; remove recoverable shards only after atomic publication.
    for rank in range(world):
        (directory/f'rank{rank}.f16').unlink();(directory/f'rank{rank}.json').unlink()
