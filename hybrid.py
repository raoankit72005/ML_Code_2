"""Run the hybrid pipeline in isolated processes to release each stage's memory."""
import argparse
import os
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from er_pipeline.common import dump_json,DEFAULTS,find_truth


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--work',type=Path,default=Path('work_hybrid'))
    parser.add_argument('--config',type=Path,default=ROOT/'configs/hybrid.json')
    parser.add_argument('--preparation-config',type=Path,default=ROOT/'configs/full_preparation.json')
    parser.add_argument('--training-config',type=Path,default=ROOT/'configs/full_gpu.json')
    parser.add_argument('--skip-test',action='store_true')
    parser.add_argument('--stage',default='all')
    args=parser.parse_args();work=args.work.resolve();work.mkdir(parents=True,exist_ok=True)
    lock_handle = None
    if args.stage == 'all':
        lock_handle = (work/'.pipeline.lock').open('a')
        try: fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another pipeline owns this work directory. Use a different --work or wait for it.')
    cfg=json.loads(args.config.read_text());prep=dict(DEFAULTS,**json.loads(args.preparation_config.read_text()))
    if cfg['encoder_mode'] not in ('frozen','lora') or cfg['device'] not in ('cpu','cuda'):
        raise ValueError('encoder_mode must be frozen/lora; device must be cpu/cuda')
    for key in ('batch_size','mini_batch_size','encode_batch_size','epochs','log_every','checkpoint_every','monitor_per_country','monitor_references','monitor_k','ann_top_k','ann_shortlist','pq_m','nlist','nprobe','ann_training_rows','cpu_threads','max_length','lora_rank'):
        if type(cfg[key]) is not int or cfg[key]<1:raise ValueError(f'{key} must be a positive integer')
    if cfg['batch_size']<2 or cfg['ann_shortlist']<cfg['ann_top_k']:
        raise ValueError('Need contrastive batch >=2 and ANN shortlist >= top k')
    from er_pipeline.modeling import file_identity,sha256,load_config
    traincfg=load_config(args.training_config)
    if not traincfg['full_data']:raise ValueError('Hybrid pipeline requires full_data=true; no implicit pair sampling')
    files=[args.input] if args.input.is_file() else sorted(args.input.rglob('*.tsv'))
    digest=hashlib.sha256()
    for p in sorted((ROOT/'src').rglob('*.py'))+[ROOT/'hybrid.py',ROOT/'gpu_worker.py',ROOT/'clean_er_data.py']:digest.update(p.read_bytes())
    signature=dict(input=[file_identity(p) for p in files],config=cfg,preparation=prep,training=traincfg,code=digest.hexdigest())
    path=work/'hybrid_manifest.json'
    state=json.loads(path.read_text()) if path.exists() else dict(signature=signature,completed=[])
    if state['signature']!=signature:raise ValueError('Inputs/config/code changed; choose a new work directory')
    dump_json(path,state)
    stages=['doctor','clean','train-index','clusters','encoder','train-embed','train-retrieve','train-block','train-label','train-features','train-export','matcher']
    if not args.skip_test:stages+=['test-index','test-embed','test-retrieve','test-block','test-features','test-export','predict','submit','audit']
    if args.stage=='all':
        for stage in stages:
            if stage in state['completed']:print('Skip completed '+stage,flush=True);continue
            command=[sys.executable,str(ROOT/'hybrid.py'),'--input',str(args.input.resolve()),'--work',str(work),
                '--config',str(args.config.resolve()),'--preparation-config',str(args.preparation_config.resolve()),
                '--training-config',str(args.training_config.resolve()),'--stage',stage]
            print('\nSTART '+stage,flush=True)
            subprocess.run(command,check=True)
            state['completed'].append(stage);dump_json(path,state)
        return
    stage=args.stage
    if stage not in stages:raise ValueError('Unknown stage')
    missing=[s for s in stages[:stages.index(stage)] if s not in state['completed']]
    if missing:raise ValueError('Run earlier stages through --stage all: '+str(missing))
    cleaned=work/'cleaned'
    from er_pipeline import indexing,blocking,labeling,features,tables,modeling,evaluation,coverage
    if stage=='doctor':
        import torch
        from er_pipeline.resources import preflight, check_disk
        check_disk(work, reserve_gb=cfg.get('disk_reserve_gb',10))
        if cfg['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('PyTorch CUDA is unavailable')
        with coverage.raw_source(args.input,'train_source1.tsv') as rows:
            if next(iter(rows),None) is None:raise ValueError('Empty training input')
        preflight(traincfg)
        if cfg['device']=='cuda':
            requested=cfg.get('gpu_workers',1)
            if torch.cuda.device_count()<requested: raise RuntimeError(f'Need {requested} visible GPUs; found {torch.cuda.device_count()}')
            for device in range(requested):
                print(f'GPU {device}: {torch.cuda.get_device_name(device)}; VRAM {torch.cuda.get_device_properties(device).total_memory/2**30:.1f} GiB',flush=True)
        workers=cfg.get('gpu_workers',1) if cfg['device']=='cuda' else cfg.get('cpu_test_workers',1)
        if workers>1:
            subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--local-addr=127.0.0.1',f'--nproc_per_node={workers}',str(ROOT/'gpu_worker.py'),'doctor','--work',str(work),'--config',str(args.config.resolve())],check=True,timeout=240)
        # Verify model revision/download and adapter targets before long preprocessing.
        from er_pipeline.encoder import load
        model=load(cfg,adapters=cfg['encoder_mode']=='lora')
        model.encode(['name: setup test | address: example | country: US'],normalize_embeddings=True)
        print('Encoder/device preflight passed',flush=True)
    elif stage=='clean':
        subprocess.run([sys.executable,str(ROOT/'clean_er_data.py'),'--input',str(args.input),'--output',str(cleaned)],check=True)
    elif stage=='clusters':
        from er_pipeline.clusters import build
        build(work/'train/index.sqlite',find_truth(cleaned),work/'clusters.sqlite',prep['validation_fraction'],cfg['seed'])
    elif stage=='encoder':
        if cfg['encoder_mode']=='lora' and (cfg.get('gpu_workers',1)>1 and cfg['device']=='cuda' or cfg.get('cpu_test_workers',1)>1):
            workers=cfg.get('gpu_workers',1) if cfg['device']=='cuda' else cfg['cpu_test_workers']
            subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--local-addr=127.0.0.1',f'--nproc_per_node={workers}',str(ROOT/'gpu_worker.py'),'encoder','--work',str(work),'--config',str(args.config.resolve())],check=True)
        else:
            from er_pipeline.encoder import train
            train(work,cfg)
    elif stage=='matcher':modeling.train(work,traincfg)
    elif stage=='predict':modeling.predict(work,'test')
    elif stage=='submit':
        _,meta=modeling.load_model(work)
        evaluation.submit(work/'test',work/'test/test_probabilities.parquet',meta['decision_threshold'])
    elif stage=='audit':coverage.validate(work,args.input)
    else:
        split,action=stage.split('-');folder=work/split;folder.mkdir(exist_ok=True)
        tfidf=work/'train/tfidf.npz'
        if action=='index':indexing.build(cleaned,folder,split,prep)
        elif action=='embed':
            workers=cfg.get('gpu_workers',1) if cfg['device']=='cuda' else cfg.get('cpu_test_workers',1)
            if workers>1:
                from er_pipeline.parallel_embed import merge
                marker=folder/'embeddings/manifest.json'
                if not marker.exists() or not json.loads(marker.read_text()).get('complete'):
                    subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--local-addr=127.0.0.1',f'--nproc_per_node={workers}',str(ROOT/'gpu_worker.py'),'embed','--work',str(work),'--split',split,'--config',str(args.config.resolve())],check=True)
                    merge(folder,workers,cfg)
            else:
                from er_pipeline.neural import embed
                embed(folder,work,cfg)
        elif action=='retrieve':
            from er_pipeline.neural import retrieve
            retrieve(folder,work,cfg)
        elif action=='block':blocking.generate_parallel(folder,tfidf,prep)
        elif action=='label':labeling.label(folder,find_truth(cleaned),prep)
        elif action=='features':features.extract(folder,tfidf,split,prep)
        elif action=='export':tables.export(folder,split,prep)

if __name__=='__main__':main()
