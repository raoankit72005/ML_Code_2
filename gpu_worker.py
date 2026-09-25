"""Launched by torchrun; never invoke concurrently with another pipeline on the same work dir."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
p=argparse.ArgumentParser();p.add_argument('stage',choices=['doctor','encoder','embed']);p.add_argument('--work',type=Path,required=True)
p.add_argument('--split',default='train');p.add_argument('--config',type=Path,required=True)
a=p.parse_args();cfg=json.loads(a.config.read_text())
if a.stage=='doctor':
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    cuda=cfg['device']=='cuda'
    if cuda:torch.cuda.set_device(rank)
    dist.init_process_group('nccl' if cuda else 'gloo',timeout=timedelta(minutes=3))
    try:
        value=torch.tensor([rank+1.],device=torch.device('cuda',rank) if cuda else 'cpu')
        dist.all_reduce(value)
        if value.item()!=world*(world+1)/2:raise RuntimeError('Collective GPU preflight failed')
        print(f'Collective preflight passed: rank={rank}, world={world}',flush=True)
    finally:dist.destroy_process_group()
elif a.stage=='encoder':
    from er_pipeline.distributed import train
    train(a.work,cfg)
else:
    from er_pipeline.parallel_embed import worker
    worker(a.work/a.split,a.work,cfg,int(os.environ.get('LOCAL_RANK',0)),int(os.environ.get('WORLD_SIZE',1)))
