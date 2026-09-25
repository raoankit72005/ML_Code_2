"""LightGBM Sequence backed by float32 files; no dense all-pair RAM array."""
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from .common import dump_json
from .resources import batch_rows, check_disk


class DiskSequence(lgb.Sequence):
    def __init__(self, path, count, nfeatures, batch_size):
        self.data = np.memmap(path, mode='r', dtype=np.float32, shape=(count, nfeatures))
        self.batch_size = batch_size
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        # LightGBM's random-row bin-construction sampler requires float64.
        return np.array(self.data[index], dtype=np.float64, copy=True)
    def close(self):
        self.data._mmap.close()


def stage_all(path, names, split, directory, batch_size, reserve_gb):
    from .modeling import matrix, file_identity
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(path)
    count = parquet.metadata.num_rows
    if not count:
        raise ValueError(f'{split} has no candidate pairs')
    required = names + ['label','dataset_split']
    if set(required) - set(parquet.schema_arrow.names):
        raise ValueError('Training table is missing features, label or dataset_split')
    signature = dict(source=file_identity(path), names=names, split=split, format_version=1)
    marker = directory/f'{split}.json'
    xp, yp = directory/f'{split}.features.f32', directory/f'{split}.labels.f32'
    expected = (count*len(names)*4, count*4)
    cached = json.loads(marker.read_text()) if marker.exists() else None
    if cached and cached['signature'] == signature and all(p.exists() and p.stat().st_size==size for p,size in zip((xp,yp),expected)):
        stats = cached['stats']
    else:
        check_disk(directory, sum(expected), reserve_gb)
        xpartial, ypartial = xp.with_suffix('.partial'), yp.with_suffix('.partial')
        seen = positives = 0
        with xpartial.open('wb') as xf, ypartial.open('wb') as yf:
            for batch in parquet.iter_batches(batch_size=batch_rows(batch_size,len(names)), columns=required):
                if any(g != split for g in batch.column('dataset_split').to_pylist()):
                    raise ValueError('Wrong dataset_split: refusing holdout leakage')
                y = batch.column('label').to_numpy(zero_copy_only=False).astype(np.float32)
                if not np.isin(y,[0,1]).all():
                    raise ValueError('Labels must be 0 or 1')
                x = matrix(batch,names)
                check_disk(directory, x.nbytes+y.nbytes, reserve_gb)
                x.tofile(xf); y.tofile(yf)
                positives += int(y.sum()); seen += len(y)
            xf.flush(); yf.flush()
            os.fsync(xf.fileno()); os.fsync(yf.fileno())
        if seen != count:
            raise ValueError('Row count changed during staging')
        xpartial.replace(xp); ypartial.replace(yp)
        stats = dict(available_pairs=count, selected_pairs=count, positive_pairs=positives,
                     negative_pairs=count-positives, sampled=False)
        dump_json(marker, dict(signature=signature, stats=stats))
    return DiskSequence(xp,count,len(names),batch_rows(batch_size,len(names))), np.memmap(yp,mode='r',dtype=np.float32), stats
