"""Single-node synchronous LoRA training. Local negatives; sample-weighted gradient all-reduce."""
import contextlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path


def rank_batch(batch, rank, world):
    # All ranks take the same number of steps. Tiny tails are replicated, not discarded.
    return batch[rank::world] if len(batch) >= 2 * world else batch


def train(work, config):
    import torch
    import torch.distributed as dist
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from . import encoder
    from .common import dump_json
    from .modeling import sha256, file_identity
    from .tracking import Tracker
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    cuda = config['device'] == 'cuda'
    if cuda: torch.cuda.set_device(local)
    device = torch.device('cuda', local) if cuda else torch.device('cpu')
    if world > 1:
        dist.init_process_group('nccl' if cuda else 'gloo', timeout=timedelta(hours=2))
    torch.set_num_threads(max(1, config['cpu_threads'] // world))
    torch.manual_seed(config['seed'])  # identical initial adapter on every rank
    model = encoder.load(config, adapters=True)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config['learning_rate'], weight_decay=.01)
    directory = work / 'encoder'; directory.mkdir(exist_ok=True)
    signature = dict(config=config, world=world, clusters=file_identity(work/'clusters.sqlite'),
                     index=file_identity(work/'train/index.sqlite'), code=sha256(__file__))
    tracker = Tracker(work, f'encoder_rank{rank}')
    def barrier():
        if world > 1: dist.barrier()
    def sum_tensor(x):
        if world > 1: dist.all_reduce(x)
        return x
    last = directory/'last.pt'; epoch0 = cursor0 = step = 0; best = -1.
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=True)
        if state['signature'] != signature: raise ValueError('Distributed checkpoint signature mismatch')
        model.load_state_dict(state['adapter'], strict=False); optimizer.load_state_dict(state['optimizer'])
        epoch0,cursor0,step,best = (state[k] for k in ('epoch','cursor','step','best'))
        rng = state['rank_rng'][rank]
        torch.set_rng_state(rng['cpu'])
        if cuda: torch.cuda.set_rng_state(rng['cuda'], device)
    else:
        torch.manual_seed(config['seed'] + rank)
    probe = None
    if rank == 0:
        probe_path = directory/'monitor.json'
        if probe_path.exists():
            raw=json.loads(probe_path.read_text()); probe=([(t,set(ids),c) for t,ids,c in raw[0]],raw[1])
        else:
            probe=encoder.monitor_set(work,config)
            dump_json(probe_path, [[[t,sorted(ids),c] for t,ids,c in probe[0]],probe[1]])
    def checkpoint(epoch, cursor, evaluate=True):
        nonlocal best
        barrier()
        value = encoder.recall(model,probe,config) if rank == 0 and evaluate else best
        score=torch.tensor([value], device=device, dtype=torch.float64)
        if world > 1: dist.broadcast(score, 0)
        improved=score.item()>best; best=max(best,score.item())
        rng={'cpu':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state(device).cpu() if cuda else None}
        states=[None]*world
        if world > 1: dist.all_gather_object(states,rng)
        else: states=[rng]
        if rank == 0:
            names={name for name,p in model.named_parameters() if p.requires_grad}
            payload=dict(adapter={k:v.detach().cpu() for k,v in model.state_dict().items() if k in names},
                         optimizer=optimizer.state_dict(),epoch=epoch,cursor=cursor,step=step,
                         best=best,signature=signature,rank_rng=states)
            def save(path):
                temp=path.with_suffix('.partial');torch.save(payload,temp);temp.replace(path)
            if improved or not (directory/'best.pt').exists(): save(directory/'best.pt')
            save(last)
            tracker.log(step,monitor_recall=score.item(),monitor_is_sampled=1,epoch=epoch)
        barrier()
    if not last.exists(): checkpoint(0,0)
    micro=config['mini_batch_size']; seen=0; started=time.monotonic()
    try:
        for epoch in range(epoch0,config['epochs']):
            for cursor,global_batch in enumerate(encoder.batches(work/'clusters.sqlite',work/'train/index.sqlite','train',config['batch_size']),1):
                if epoch == epoch0 and cursor <= cursor0: continue
                batch=rank_batch(global_batch,rank,world)
                model.train()
                while True:
                    optimizer.zero_grad(set_to_none=True); failed=0; loss=None; features=None
                    try:
                        features=[{k:v.to(device) for k,v in model.tokenize(list(texts)).items()} for texts in zip(*batch)]
                        objective=CachedMultipleNegativesRankingLoss(model, mini_batch_size=micro)
                        with torch.autocast('cuda',dtype=torch.bfloat16) if cuda and torch.cuda.is_bf16_supported() else contextlib.nullcontext():
                            loss=objective(features,torch.empty(0,device=device));loss.backward()
                    except torch.cuda.OutOfMemoryError:
                        failed=1
                    failures=sum_tensor(torch.tensor([failed],device=device)).item()
                    if not failures: break
                    optimizer.zero_grad(set_to_none=True);loss=None;features=None
                    if cuda: torch.cuda.empty_cache()
                    if micro <= 1: raise RuntimeError('OOM even at activation mini-batch 1; checkpoint retained')
                    micro=max(1,micro//2);tracker.log(step,oom_retry=1,activation_mini_batch=micro)
                stats=sum_tensor(torch.tensor([float(loss.detach())*len(batch),len(batch)],device=device,dtype=torch.float64))
                # Gradient synchronization is outside cached backward, avoiding DDP's repeated-forward hooks.
                for p in params:
                    if p.grad is None: p.grad=torch.zeros_like(p)
                    p.grad.mul_(len(batch));sum_tensor(p.grad);p.grad.div_(stats[1])
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                if not torch.isfinite(stats).all(): raise RuntimeError('Non-finite distributed loss')
                optimizer.step();step+=1;seen+=len(global_batch)
                if step%config['log_every']==0:
                    tracker.log(step,loss=(stats[0]/stats[1]).item(),epoch=epoch+1,world_size=world,
                                pairs_processed=seen,pairs_per_second=seen/max(.001,time.monotonic()-started),
                                activation_mini_batch=micro,local_batch=len(batch))
                if step%config['checkpoint_every']==0: checkpoint(epoch,cursor)
            checkpoint(epoch+1,0)
        if rank==0:
            dump_json(directory/'complete.json',dict(signature=signature,mode='lora',best_monitor_recall=best,
                selected_sha256=sha256(directory/'best.pt'),monitor_note='Sampled monitor; full-reference metrics are computed later'))
        barrier()
    finally:
        tracker.close()
        if world>1: dist.destroy_process_group()
