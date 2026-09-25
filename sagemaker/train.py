"""SageMaker BYOC entry point. One instance, multiple local GPUs. No job submission here."""
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def discover(channel):
    zips=sorted(channel.rglob('*.zip'))
    tsvs=list(channel.rglob('train_source1.tsv'))
    if len(zips)==1 and not tsvs:return zips[0]
    if not zips and len(tsvs)==1:return channel
    raise ValueError('Dataset channel must contain exactly one original ZIP or one original TSV dataset, not both')


def copy_outputs(work,model,out):
    model.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    shutil.copytree(work/'model',model/'matcher',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('data_cache','*.f32','*.parquet','*.partial','training_snapshot.txt'))
    (model/'encoder').mkdir(exist_ok=True)
    for name in ('best.pt','complete.json'):
        if (work/'encoder'/name).exists():shutil.copy2(work/'encoder'/name,model/'encoder'/name)
    shutil.copy2(work/'train/tfidf.npz',model/'tfidf.npz')
    shutil.copytree(ROOT/'configs',model/'configs',dirs_exist_ok=True)
    shutil.copytree(ROOT/'src',model/'src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('hybrid.py','gpu_worker.py','clean_er_data.py','requirements.txt','requirements-hybrid.txt','README.md'):
        shutil.copy2(ROOT/name,model/name)
    shutil.copytree(work/'logs',out/'logs',dirs_exist_ok=True)
    for path in (work/'test').glob('*.tsv'):shutil.copy2(path,out/path.name)
    for folder in ('train','test','model'):
        for path in (work/folder).glob('*.json'):
            dest=out/'reports'/folder;dest.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest/path.name)
    if (ROOT/'environment-resolved.txt').exists():shutil.copy2(ROOT/'environment-resolved.txt',model/'environment-resolved.txt')


def main():
    # SageMaker appends "train" to the image entrypoint; this script needs no positional arguments.
    channel=Path(os.environ.get('SM_CHANNEL_DATASET','/opt/ml/input/data/dataset'))
    data=discover(channel)
    work=Path(os.environ.get('ER_WORK_DIR','/opt/ml/work'));work.mkdir(parents=True,exist_ok=True)
    for key,folder in [('TMPDIR','tmp'),('HF_HOME','huggingface'),('SQLITE_TMPDIR','tmp')]:
        target=work/folder;target.mkdir(exist_ok=True);os.environ[key]=str(target)
    resource=Path('/opt/ml/input/config/resourceconfig.json')
    if resource.exists() and len(json.loads(resource.read_text()).get('hosts',[]))!=1:
        raise ValueError('This job supports one multi-GPU instance, not multiple instances')
    checkpoint=Path('/opt/ml/checkpoints');checkpoint.mkdir(parents=True,exist_ok=True)
    # Save portable encoder checkpoints to S3; these are not a complete pipeline resume snapshot.
    enc=checkpoint/'encoder';enc.mkdir(exist_ok=True)
    if not (work/'encoder').exists():(work/'encoder').symlink_to(enc,target_is_directory=True)
    command=[sys.executable,'-u',str(ROOT/'hybrid.py'),'--input',str(data),'--work',str(work)]
    print('Launching: '+' '.join(command),flush=True)
    child=subprocess.Popen(command,cwd=ROOT,start_new_session=True)
    def stop(signum,frame):
        print('Termination requested; preserving completed checkpoints.',flush=True)
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,stop)
    status=child.wait()
    out=Path(os.environ.get('SM_OUTPUT_DATA_DIR','/opt/ml/output/data'))
    out.mkdir(parents=True,exist_ok=True)
    if status:
        failure=Path('/opt/ml/output/failure');failure.parent.mkdir(parents=True,exist_ok=True)
        failure.write_text(f'Pipeline exited {status}. See CloudWatch logs; last completed encoder checkpoints are preserved.')
        if (work/'logs').exists():shutil.copytree(work/'logs',out/'logs',dirs_exist_ok=True)
        raise SystemExit(status if status>0 else 1)
    copy_outputs(work,Path(os.environ.get('SM_MODEL_DIR','/opt/ml/model')),out)
    print('COMPLETE: matching_results.tsv and candidate_pairs.tsv exported to output/data; model bundle exported to model.',flush=True)

if __name__=='__main__':main()
