"""Resource estimates and early failures, never silent row sampling."""
import json
import os
import shutil
import subprocess
from pathlib import Path

GIB = 1024 ** 3


def available_ram():
    """Linux MemAvailable limited by cgroup headroom (SageMaker/container aware)."""
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.strip().split()[0]) * 1024
    available = values['MemAvailable']
    # Check root plus current nested cgroup, including all ancestor limits.
    roots = [Path('/sys/fs/cgroup'), Path('/sys/fs/cgroup/memory')]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        base = Path('/sys/fs/cgroup/memory') if 'memory' in controllers.split(',') else Path('/sys/fs/cgroup')
        child = base / relative.lstrip('/')
        if '..' not in child.parts:
            roots.extend([child] + [p for p in child.parents if p == base or base in p.parents])
    for root in set(roots):
        for limit, used in [('memory.max','memory.current'), ('memory.limit_in_bytes','memory.usage_in_bytes')]:
            try:
                available = min(available, max(0, int((root/limit).read_text()) - int((root/used).read_text())))
            except (OSError, ValueError):
                pass
    return available


def gpu_memory(device_id=0):
    """Map CUDA_VISIBLE_DEVICES to the physical GPU for NVIDIA memory inspection."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    physical = str(device_id)
    if visible:
        choices = visible.split(',')
        if device_id >= len(choices):
            raise ValueError('gpu_device_id is outside CUDA_VISIBLE_DEVICES')
        physical = choices[device_id].strip()
    try:
        result = subprocess.run(['nvidia-smi', '-i', physical,
            '--query-gpu=name,memory.free,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=20)
        name, free, total = result.stdout.strip().split(',')
        return dict(name=name.strip(), free_bytes=int(free)*1024**2, total_bytes=int(total)*1024**2)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError('Cannot inspect NVIDIA GPU. Use a GPU SageMaker app with a working nvidia-smi.') from exc


def check_disk(path, required=0, reserve_gb=2):
    free = shutil.disk_usage(path).free
    if free < required + reserve_gb * GIB:
        raise RuntimeError(f'Insufficient disk: {free/GIB:.1f} GiB free; need at least {(required/GIB)+reserve_gb:.1f} GiB. Expand the work volume; no data was sampled.')
    return free


def batch_rows(requested, nfeatures, fraction=0.01):
    # Includes Arrow, Python conversion and float64 Sequence copies.
    budget = int(available_ram() * fraction)
    if budget < max(1, nfeatures)*64:
        raise RuntimeError('Insufficient available RAM even for one loading row')
    return max(1, min(requested, budget // (max(1, nfeatures)*64)))


def training_plan(train_rows, valid_rows, nfeatures, config, work):
    """Conservative heuristic, NOT a proof of peak native/CUDA allocation."""
    total = train_rows + valid_rows
    # Bins + row indices, gradients, CUDA work buffers, labels and construction.
    bin_bytes = 1 if config['max_bin'] <= 255 else 2
    raw_bytes = total * (4*nfeatures + 4)
    resident = total * (2*nfeatures*bin_bytes + 128)
    hist = config['num_leaves'] * nfeatures * config['max_bin'] * 32 * config['num_threads']
    ram_need = int(1.5 * (resident + hist) + 512*1024**2)
    gpu_need = int(1.5 * (total*(2*nfeatures*bin_bytes + 96) + hist) + GIB)
    ram_free = available_ram()
    plan = dict(train_pairs=train_rows, validation_pairs=valid_rows, features=nfeatures,
        estimated_ram_bytes=ram_need, available_ram_bytes=ram_free,
        staging_disk_bytes=raw_bytes, device_type=config['device_type'],
        batch_size=batch_rows(config['batch_size'], nfeatures),
        note='Heuristic with headroom; native library peaks and concurrent workloads can differ. All non-holdout training pairs are retained.')
    errors = []
    if ram_need > ram_free * config['ram_fraction']:
        errors.append(f'RAM estimate {ram_need/GIB:.1f} GiB exceeds permitted {ram_free*config["ram_fraction"]/GIB:.1f} GiB')
    if config['device_type'] == 'cuda':
        gpu = gpu_memory(config['gpu_device_id'])
        plan.update(gpu=gpu, estimated_gpu_bytes=gpu_need)
        if gpu_need > gpu['free_bytes'] * config['gpu_memory_fraction']:
            errors.append(f'GPU estimate {gpu_need/GIB:.1f} GiB exceeds permitted {gpu["free_bytes"]*config["gpu_memory_fraction"]/GIB:.1f} GiB')
    plan['errors'] = errors
    from .common import dump_json
    dump_json(Path(work)/'resource_plan.json', plan)
    print(json.dumps(plan, indent=2), flush=True)
    if errors:
        raise RuntimeError('; '.join(errors) + '. Use a larger instance/GPU. See resource_plan.json. Full-data training will not silently sample rows.')
    return plan


def preflight(config):
    """Exercise the actual installed backend before costly preprocessing."""
    import lightgbm as lgb
    import numpy as np
    result = dict(lightgbm=lgb.__version__, device_type=config['device_type'],
                  available_ram_gib=round(available_ram()/GIB, 2))
    if config['device_type'] == 'cuda':
        result['gpu'] = gpu_memory(config['gpu_device_id'])
    rng = np.random.default_rng(42)
    x = rng.normal(size=(256, 4))
    try:
        lgb.train(dict(objective='binary', verbosity=-1, device_type=config['device_type'],
            gpu_device_id=config['gpu_device_id'], num_threads=min(2,config['num_threads']),
            max_bin=config['max_bin'], min_data_in_leaf=5),
            lgb.Dataset(x, label=(x[:,0]>0).astype(int)), num_boost_round=2)
    except lgb.basic.LightGBMError as exc:
        raise RuntimeError('Requested LightGBM backend failed. Run scripts/setup_gpu.sh in a CUDA development environment; no CPU fallback was used.') from exc
    print(json.dumps(result, indent=2), flush=True)
    return result
