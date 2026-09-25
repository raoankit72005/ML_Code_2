"""LoRA Siamese encoder, streamed unique-cluster batches, resumable training."""
import contextlib
import json
import random
import time
from pathlib import Path

import numpy as np

from .common import connect, lookup, dump_json
from .tracking import Tracker


def text(record):
    return f"name: {record['name_basic']} | address: {record['address_normalized']} | country: {record['country_key']}"


def load(config, checkpoint=None, adapters=False):
    import torch
    from sentence_transformers import SentenceTransformer
    if config['device']=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but PyTorch cannot use GPU; run the setup and doctor commands')
    model=SentenceTransformer(config['model_id'],revision=config['revision'],device=config['device'])
    model.max_seq_length=config['max_length']
    if adapters:
        from peft import LoraConfig
        model.add_adapter(LoraConfig(r=config['lora_rank'],lora_alpha=2*config['lora_rank'],
            lora_dropout=0.0,target_modules=['query','value'],bias='none'))
        if not any(p.requires_grad for p in model.parameters()):raise RuntimeError('No trainable LoRA parameters')
    if checkpoint:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        incompatible=model.load_state_dict(state['adapter'],strict=False)
        if incompatible.unexpected_keys:raise ValueError('Adapter checkpoint architecture mismatch')
    return model


def batches(clusters, records, split, batch_size):
    """All positives once/epoch. Round-robin positive ordinals avoid adjacent same-anchor pairs."""
    conn=connect(clusters,readonly=True);refs=connect(records,readonly=True)
    pending=[];seen=set();country_now=None
    def ready():
        if len(pending)==1:
            other=conn.execute('SELECT p.sid,p.eid FROM pairs p JOIN queries q USING(sid) WHERE q.split=? AND q.cluster!=? ORDER BY q.country!=? LIMIT 1',(split,next(iter(seen)),country_now)).fetchone()
            if not other:raise ValueError('Contrastive training needs at least two training clusters')
            # Replay a different-cluster positive as a valid negative partner; never drop tail rows.
            return pending+[(text(lookup(refs,other[0])),text(lookup(refs,other[1])))]
        return pending
    sql='''SELECT p.sid,p.eid,q.cluster,q.country FROM pairs p JOIN queries q USING(sid)
           WHERE q.split=? ORDER BY q.country,p.ordinal,q.shuffle,p.sid,p.eid'''
    try:
        for sid,eid,cluster,country in conn.execute(sql,(split,)):
            if pending and (cluster in seen or country!=country_now or len(pending)>=batch_size):
                yield ready();pending=[];seen=set()
            pending.append((text(lookup(refs,sid)),text(lookup(refs,eid))))
            seen.add(cluster);country_now=country
        if pending:yield ready()
    finally:conn.close();refs.close()


def monitor_set(work,config):
    conn=connect(work/'clusters.sqlite',readonly=True);refs=connect(work/'train/index.sqlite',readonly=True)
    queries=[];wanted=set()
    for (country,) in conn.execute("SELECT DISTINCT country FROM queries WHERE split='validation'"):
        for sid,raw in conn.execute("SELECT sid,ids FROM queries WHERE split='validation' AND country=? AND ids!='[]' ORDER BY shuffle,sid LIMIT ?",(country,config['monitor_per_country'])):
            true=set(json.loads(raw));wanted.update(true)
            queries.append((text(lookup(refs,sid)),true,country))
    # Deterministic reservoir drawn only from provided reference records.
    rng=random.Random(config['seed']);sample=[]
    for n,(eid,) in enumerate(refs.execute('SELECT entity_id FROM records WHERE source IN (2,3) ORDER BY rid'),1):
        if len(sample)<config['monitor_references']:sample.append(eid)
        else:
            j=rng.randrange(n)
            if j<len(sample):sample[j]=eid
    wanted.update(sample)
    documents=[(eid,text(lookup(refs,eid)),lookup(refs,eid)['country_key']) for eid in sorted(wanted)]
    conn.close();refs.close()
    if not queries:raise ValueError('No positive validation queries for encoder monitoring')
    return queries,documents


def recall(model,probe,config):
    queries,documents=probe
    model.eval()
    docs=model.encode([x[1] for x in documents],batch_size=config['encode_batch_size'],
        normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    qvec=model.encode([x[0] for x in queries],batch_size=config['encode_batch_size'],
        normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    total=hits=0
    for vector,(_,truth,country) in zip(qvec,queries):
        valid=np.array([i for i,d in enumerate(documents) if not country or not d[2] or country==d[2]],dtype=np.int64)
        if len(valid):
            scores=docs[valid]@vector;k=min(config['monitor_k'],len(valid))
            indices=valid[np.argpartition(scores,len(scores)-k)[-k:]]
            hits+=len(truth & {documents[i][0] for i in indices})
        total+=len(truth)
    return hits/max(1,total)


def save_checkpoint(path,model,optimizer,epoch,cursor,step,best,signature):
    import torch
    names={name for name,p in model.named_parameters() if p.requires_grad}
    adapter={k:v.detach().cpu() for k,v in model.state_dict().items() if k in names}
    value=dict(adapter=adapter,optimizer=optimizer.state_dict(),epoch=epoch,cursor=cursor,
        step=step,best=best,signature=signature,rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temporary=path.with_suffix('.partial');torch.save(value,temporary);temporary.replace(path)


def train(work,config):
    import torch
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from .modeling import file_identity,sha256
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    random.seed(config['seed']);np.random.seed(config['seed'])
    directory=work/'encoder';directory.mkdir(exist_ok=True)
    signature=dict(config=config,clusters=file_identity(work/'clusters.sqlite'),
        index=file_identity(work/'train/index.sqlite'),code=sha256(__file__))
    marker=directory/'complete.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['signature']!=signature:raise ValueError('Encoder inputs/config changed: use a new work folder')
        if old.get('selected_sha256') and sha256(directory/'best.pt')!=old['selected_sha256']:
            raise ValueError('Selected encoder checkpoint was changed')
        return
    if config['encoder_mode']=='frozen':
        dump_json(marker,dict(signature=signature,mode='frozen'));return
    tracker=Tracker(work,'encoder')
    model=load(config,adapters=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=.01)
    loss_function=CachedMultipleNegativesRankingLoss(model,mini_batch_size=config['mini_batch_size'])
    probe_path=directory/'monitor.json'
    if probe_path.exists():
        raw=json.loads(probe_path.read_text());probe=([(t,set(ids),c) for t,ids,c in raw[0]],raw[1])
    else:
        probe=monitor_set(work,config)
        dump_json(probe_path,[[[t,sorted(ids),c] for t,ids,c in probe[0]],probe[1]])
    step=start_epoch=start_cursor=0;best=-1.
    last=directory/'last.pt'
    if last.exists():
        state=torch.load(last,map_location=config['device'],weights_only=True)
        if state['signature']!=signature:raise ValueError('Checkpoint does not match this run')
        model.load_state_dict(state['adapter'],strict=False);optimizer.load_state_dict(state['optimizer'])
        start_epoch,start_cursor,step,best=(state[k] for k in ('epoch','cursor','step','best'))
        torch.set_rng_state(state['rng'].cpu())
        if state['cuda_rng']:torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    else:
        best=recall(model,probe,config)
        save_checkpoint(directory/'best.pt',model,optimizer,0,0,0,best,signature)
        tracker.log(0,monitor_recall=best,monitor_is_sampled=1)
    started=time.monotonic();seen=0
    try:
        for epoch in range(start_epoch,config['epochs']):
            cursor=0
            for cursor,batch in enumerate(batches(work/'clusters.sqlite',work/'train/index.sqlite','train',config['batch_size']),1):
                if epoch==start_epoch and cursor<=start_cursor:continue
                # A singleton contrastive batch has no negative; refuse rather than silently drop training rows.
                if len(batch)<2:
                    raise ValueError('An entity-aware training batch has fewer than 2 clusters. Increase training data or adjust batching; no positives were silently dropped.')
                model.train();optimizer.zero_grad(set_to_none=True)
                features=[{k:v.to(config['device']) for k,v in model.tokenize(list(texts)).items()} for texts in zip(*batch)]
                mixed=config['device']=='cuda' and torch.cuda.is_bf16_supported()
                context=torch.autocast('cuda',dtype=torch.bfloat16) if mixed else contextlib.nullcontext()
                while True:
                    try:
                        with context:
                            loss=loss_function(features,torch.empty(0,device=config['device']))
                            loss.backward()
                        break
                    except torch.cuda.OutOfMemoryError:
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        current=loss_function.mini_batch_size
                        if current<=1:raise RuntimeError('Encoder cannot fit even one activation example. Last saved checkpoint is retained.')
                        loss_function=CachedMultipleNegativesRankingLoss(model,mini_batch_size=max(1,current//2))
                        tracker.log(step,oom_retry=1,activation_mini_batch=loss_function.mini_batch_size)
                        context=torch.autocast('cuda',dtype=torch.bfloat16) if mixed else contextlib.nullcontext()
                if not torch.isfinite(loss):raise RuntimeError('Non-finite encoder loss')
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step();step+=1;seen+=len(batch)
                if step%config['log_every']==0:
                    tracker.log(step,loss=float(loss.detach()),epoch=epoch+1,learning_rate=optimizer.param_groups[0]['lr'],
                        pairs_per_second=seen/max(1e-6,time.monotonic()-started),pairs_processed_this_session=seen)
                if step%config['checkpoint_every']==0:
                    value=recall(model,probe,config);tracker.log(step,monitor_recall=value,monitor_is_sampled=1)
                    if value>best:
                        best=value;save_checkpoint(directory/'best.pt',model,optimizer,epoch,cursor,step,best,signature)
                    save_checkpoint(last,model,optimizer,epoch,cursor,step,best,signature)
            value=recall(model,probe,config);tracker.log(step,monitor_recall=value,epoch=epoch+1,monitor_is_sampled=1)
            if value>best:
                best=value;save_checkpoint(directory/'best.pt',model,optimizer,epoch+1,0,step,best,signature)
            save_checkpoint(last,model,optimizer,epoch+1,0,step,best,signature)
        dump_json(marker,dict(signature=signature,mode='lora',best_monitor_recall=best,
            selected_sha256=sha256(directory/'best.pt'),monitor_note='Sampled reference pool; not final full-pool recall or leaderboard estimate'))
    finally:tracker.close()
