"""Local JSONL and TensorBoard metrics; no external tracking account needed."""
import json
import time
from pathlib import Path

class Tracker:
    def __init__(self, work, stage):
        self.stage=stage
        self.directory=Path(work)/'logs'; self.directory.mkdir(parents=True,exist_ok=True)
        self.file=(self.directory/f'{stage}.jsonl').open('a',buffering=1)
        self.started=time.monotonic()
        from torch.utils.tensorboard import SummaryWriter
        self.writer=SummaryWriter(str(self.directory/'tensorboard'/stage),flush_secs=15)
    def log(self, step, **metrics):
        import psutil
        metrics['rss_gib']=psutil.Process().memory_info().rss/1024**3
        metrics['ram_available_gib']=psutil.virtual_memory().available/1024**3
        try:
            import torch
            if torch.cuda.is_available():
                metrics['gpu_allocated_gib']=torch.cuda.memory_allocated()/1024**3
                metrics['gpu_reserved_gib']=torch.cuda.memory_reserved()/1024**3
                metrics['gpu_peak_gib']=torch.cuda.max_memory_allocated()/1024**3
        except ImportError:
            pass
        row=dict(stage=self.stage,step=step,time=time.time(),elapsed_seconds=time.monotonic()-self.started,**metrics)
        self.file.write(json.dumps(row,allow_nan=False)+'\n')
        for key,value in metrics.items():
            if isinstance(value,(int,float)): self.writer.add_scalar(key,value,step)
        self.writer.flush()
        print(json.dumps(row),flush=True)
    def close(self):
        self.file.close(); self.writer.close()
