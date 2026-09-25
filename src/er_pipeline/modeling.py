"""CPU/CUDA LightGBM with full-data disk loading and streaming scoring."""
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .common import dump_json
from . import evaluation

DEFAULT_TRAINING = dict(
    full_data=False, device_type="cpu", gpu_device_id=0, ram_fraction=0.65,
    gpu_memory_fraction=0.75, disk_reserve_gb=5, histogram_pool_size=256,
    seed=42, num_threads=4, num_boost_round=600, early_stopping_rounds=50,
    learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, max_bin=63,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, lambda_l2=2.0,
    max_train_pairs=1000000, max_early_stopping_pairs=200000, batch_size=20000,
    thresholds=[round(i / 100, 2) for i in range(5, 100, 5)] + [0.925, 0.975, 0.99, 0.995, 1.0])
PROBABILITY_SCHEMA = pa.schema([
    ('source1_entity_id', pa.string()), ('candidate_entity_id', pa.string()),
    ('match_probability', pa.float64())])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def load_config(path=None, overrides=None):
    config = dict(DEFAULT_TRAINING)
    updates = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    updates.update({k: v for k, v in (overrides or {}).items() if v is not None})
    unknown = updates.keys() - config.keys()
    if unknown:
        raise ValueError(f'Unknown training configuration: {sorted(unknown)}')
    config.update(updates)
    positive_ints = ['num_threads', 'num_boost_round', 'early_stopping_rounds', 'num_leaves',
        'min_data_in_leaf', 'max_bin', 'max_train_pairs', 'max_early_stopping_pairs', 'batch_size']
    for key in positive_ints:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be an integer > 0')
    if type(config['full_data']) is not bool:
        raise ValueError('full_data must be boolean')
    if config['device_type'] not in ('cpu', 'cuda'):
        raise ValueError('device_type must be cpu or cuda')
    if type(config['gpu_device_id']) is not int or config['gpu_device_id'] < 0:
        raise ValueError('gpu_device_id must be a nonnegative integer')
    for key in ('ram_fraction', 'gpu_memory_fraction'):
        if not 0 < config[key] <= 0.9:
            raise ValueError(f'{key} must be in (0, 0.9]')
    for key in ('disk_reserve_gb', 'histogram_pool_size'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(config['bagging_freq']) is not int or config['bagging_freq'] < 0:
        raise ValueError('bagging_freq must be a nonnegative integer')
    for key in ('feature_fraction', 'bagging_fraction', 'learning_rate'):
        if not 0 < config[key] <= 1:
            raise ValueError(f'{key} must be in (0,1]')
    if not math.isfinite(config['lambda_l2']) or config['lambda_l2'] < 0:
        raise ValueError('lambda_l2 must be finite and >= 0')
    if config['num_leaves'] < 2 or config['max_bin'] < 2:
        raise ValueError('num_leaves and max_bin must be >= 2')
    if not config['thresholds'] or any(not math.isfinite(t) or not 0 <= t <= 1 for t in config['thresholds']):
        raise ValueError('thresholds must be a nonempty list of finite values in [0,1]')
    config['thresholds'] = sorted(set(config['thresholds']))
    return config


def feature_names(work):
    data = json.loads((Path(work) / 'feature_columns.json').read_text())
    names = data['features']
    forbidden = {'label', 'dataset_split', 'source1_entity_id', 'candidate_entity_id'}
    if not names or len(names) != len(set(names)) or forbidden.intersection(names):
        raise ValueError('Invalid model input list: IDs, labels and dataset_split cannot be features')
    return names


def matrix(batch, names):
    result = np.empty((batch.num_rows, len(names)), dtype=np.float32)
    for i, name in enumerate(names):
        column = batch.column(batch.schema.get_field_index(name))
        if not (pa.types.is_integer(column.type) or pa.types.is_floating(column.type)):
            raise ValueError(f'Non-numeric feature: {name}')
        result[:, i] = column.to_numpy(zero_copy_only=False)
    if np.isinf(result).any():
        raise ValueError('Infinite feature values found; NaN is allowed but infinity is not')
    return result


def load_training_rows(path, names, cap, seed, expected_split, batch_size):
    """Uniform sampling without replacement over pair rows; never mix S1 splits.

    Only training / early-stopping monitor rows are capped. Prediction and final
    threshold selection ALWAYS use every validation candidate and query.
    """
    file = pq.ParquetFile(path)
    required = names + ['label', 'dataset_split']
    missing = set(required) - set(file.schema_arrow.names)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    total = file.metadata.num_rows
    if not total:
        raise ValueError(f'{path} is empty: need candidate pairs in both train and validation')
    count = min(total, cap)
    chosen = np.sort(np.random.default_rng(seed).choice(total, size=count, replace=False)) if count < total else None
    x = np.empty((count, len(names)), dtype=np.float32)
    y = np.empty(count, dtype=np.float32)
    offset = cursor = 0
    for batch in file.iter_batches(batch_size=batch_size, columns=required):
        groups = batch.column(batch.schema.get_field_index('dataset_split')).to_pylist()
        if any(g != expected_split for g in groups):
            raise ValueError(f'{path}: wrong dataset_split; refusing train/validation mixing')
        if chosen is None:
            indices = None
            length = batch.num_rows
        else:
            end = np.searchsorted(chosen, offset + batch.num_rows)
            indices = chosen[cursor:end] - offset
            length = len(indices)
        if length:
            selected = batch if indices is None else batch.take(pa.array(indices, type=pa.int64()))
            x[cursor:cursor+length] = matrix(selected, names)
            labels = selected.column(selected.schema.get_field_index('label')).to_numpy(zero_copy_only=False)
            if not np.isin(labels, [0, 1]).all():
                raise ValueError('Labels must be 0 or 1')
            y[cursor:cursor+length] = labels
            cursor += length
        offset += batch.num_rows
    if cursor != count:
        raise ValueError('Parquet row count changed while reading')
    summary = dict(available_pairs=total, selected_pairs=count, positive_pairs=int(y.sum()),
                   negative_pairs=int(count-y.sum()), sampled=count < total)
    return x, y, summary


def _predict_file(booster, source, destination, names, batch_size, num_threads):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Probability output cannot overwrite its input features')
    from .resources import batch_rows, check_disk
    batch_size = batch_rows(batch_size, len(names))
    file = pq.ParquetFile(source)
    if set(names + ['source1_entity_id', 'candidate_entity_id']) - set(file.schema_arrow.names):
        raise ValueError('Prediction table is missing model features or pair IDs')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    count = 0
    with pq.ParquetWriter(partial, PROBABILITY_SCHEMA, compression='zstd') as writer:
        for batch in file.iter_batches(batch_size=batch_size, columns=['source1_entity_id', 'candidate_entity_id'] + names):
            probabilities = booster.predict(matrix(batch, names), num_threads=num_threads)
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError('Model produced invalid probabilities')
            output = pa.Table.from_arrays([
                batch.column(batch.schema.get_field_index('source1_entity_id')),
                batch.column(batch.schema.get_field_index('candidate_entity_id')),
                pa.array(probabilities, type=pa.float64())], schema=PROBABILITY_SCHEMA)
            check_disk(destination.parent, output.nbytes)
            writer.write_table(output)
            count += batch.num_rows
    if count != file.metadata.num_rows:
        raise ValueError('Not every candidate received a probability')
    partial.replace(destination)
    return count


def train(root, config, model_dir=None):
    root = Path(root).resolve()
    work = root / 'train'
    model_dir = Path(model_dir).resolve() if model_dir else root / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)
    names = feature_names(work)
    source = work / 'train_features.parquet'
    validation = work / 'validation_features.parquet'
    signature = dict(config=config, inputs=[file_identity(p) for p in
        (source, validation, work/'query_labels.tsv.gz', work/'feature_columns.json')],
        tfidf_sha256=sha256(work/'tfidf.npz'), code_sha256=sha256(__file__), loader_sha256=sha256(Path(__file__).with_name('disk_training.py')),
        resources_sha256=sha256(Path(__file__).with_name('resources.py')))
    metadata_path = model_dir / 'model_metadata.json'
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing.get('signature') != signature:
            raise ValueError('Training inputs/config/code changed. Use a new --model-dir or --work folder.')
        if existing.get('status') == 'complete':
            load_model(root, model_dir)
            for filename in ('validation_probabilities.parquet', 'validation_f05.json', 'feature_importance.tsv'):
                if not (model_dir/filename).exists():
                    raise ValueError('Completed training artifact is missing; choose a new --model-dir')
            print(f'Skip completed training: {model_dir}', flush=True)
            return existing
    else:
        dump_json(metadata_path, dict(status='running', signature=signature))
    from .resources import preflight, training_plan
    from .disk_training import stage_all
    preflight(config)
    if config['full_data']:
        plan = training_plan(pq.ParquetFile(source).metadata.num_rows,
            pq.ParquetFile(validation).metadata.num_rows, len(names), config, model_dir)
        print('Loading ALL train and holdout pairs through disk-backed Sequence...', flush=True)
        x, y, train_stats = stage_all(source, names, 'train', model_dir/'data_cache',
            plan['batch_size'], config['disk_reserve_gb'])
        xv, yv, monitor_stats = stage_all(validation, names, 'validation', model_dir/'data_cache',
            plan['batch_size'], config['disk_reserve_gb'])
    else:
        print('Loading explicitly capped baseline arrays...', flush=True)
        x, y, train_stats = load_training_rows(source, names, config['max_train_pairs'], config['seed'], 'train', config['batch_size'])
        xv, yv, monitor_stats = load_training_rows(validation, names, config['max_early_stopping_pairs'], config['seed']+1, 'validation', config['batch_size'])
    if not train_stats['positive_pairs'] or not train_stats['negative_pairs']:
        raise ValueError('Training needs both positive and negative pairs')
    params = {k: config[k] for k in ('learning_rate', 'num_leaves', 'min_data_in_leaf', 'max_bin',
        'feature_fraction', 'bagging_fraction', 'bagging_freq', 'lambda_l2', 'num_threads', 'seed')}
    params.update(objective='binary', metric='binary_logloss', device_type=config['device_type'],
        gpu_device_id=config['gpu_device_id'], histogram_pool_size=config['histogram_pool_size'],
        verbosity=-1,
        data_random_seed=config['seed'], feature_fraction_seed=config['seed'], bagging_seed=config['seed'])
    if config['device_type'] == 'cpu':
        params.update(deterministic=True, force_col_wise=True)
    training_set = lgb.Dataset(x, label=y, feature_name=names, free_raw_data=True)
    validation_set = lgb.Dataset(xv, label=yv, reference=training_set, feature_name=names, free_raw_data=True)
    from .tracking import Tracker
    tracker=Tracker(root,'lightgbm')
    def track(env):
        step=env.iteration+1
        if step%10==0:
            metrics={f'{data}_{metric}':float(value) for data,metric,value,*rest in env.evaluation_result_list}
            tracker.log(step,**metrics)
        if step%100==0:
            env.model.save_model(str(model_dir/'training_snapshot.txt'))
    history = {}
    print(f'LightGBM: {len(y):,} training pairs; {len(yv):,} early-stopping pairs', flush=True)
    booster = lgb.train(params, training_set, num_boost_round=config['num_boost_round'],
        valid_sets=[validation_set], valid_names=['validation_monitor'], callbacks=[
            lgb.early_stopping(config['early_stopping_rounds'], first_metric_only=True, verbose=True),
            lgb.log_evaluation(10), lgb.record_evaluation(history),track])
    best_iteration = booster.best_iteration or booster.current_iteration()
    model_path = model_dir/'lightgbm_model.txt'
    temporary = model_dir/'lightgbm_model.txt.partial'
    booster.save_model(str(temporary), num_iteration=best_iteration)
    temporary.replace(model_path)
    dump_json(model_dir/'learning_curve.json', history)
    with (model_dir/'feature_importance.tsv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['feature', 'gain', 'splits'])
        data = zip(names, booster.feature_importance('gain', iteration=best_iteration),
                   booster.feature_importance('split', iteration=best_iteration))
        writer.writerows(sorted(data, key=lambda row: -row[1]))
    booster.free_dataset()
    del training_set, validation_set
    if config['full_data']:
        x.close(); xv.close()
        y._mmap.close(); yv._mmap.close()
    del x, y, xv, yv, booster
    gc.collect()
    booster = lgb.Booster(model_file=str(model_path))
    probability_path = model_dir/'validation_probabilities.parquet'
    count = _predict_file(booster, validation, probability_path, names, config['batch_size'], config['num_threads'])
    result = evaluation.evaluate(work, probability_path, config['thresholds'])
    # Preserve the exact metric selected for this model, independently of later evaluations.
    dump_json(model_dir/'validation_f05.json', result)
    metadata = dict(status='complete', signature=signature, features=names,
        lightgbm_version=lgb.__version__, model_sha256=sha256(model_path),
        best_iteration=best_iteration, decision_threshold=result['best']['threshold'],
        validation_macro_f05=result['best']['macro_f05'], train_rows=train_stats,
        early_stopping_rows=monitor_stats, final_validation_pairs=count,
        validation_query_counts=result['query_counts'],
        model_note='Early stopping uses binary log loss; threshold uses full validation macro F0.5. This is a tuned holdout score, not a final-test score. Model is not refit on validation.')
    dump_json(metadata_path, metadata)
    tracker.log(best_iteration,macro_f05=result['best']['macro_f05'],selected_threshold=result['best']['threshold'])
    tracker.close()
    print(f'Model saved: {model_path}', flush=True)
    return metadata


def load_model(root, model_dir=None):
    root = Path(root).resolve()
    model_dir = Path(model_dir).resolve() if model_dir else root/'model'
    metadata = json.loads((model_dir/'model_metadata.json').read_text())
    if metadata.get('status') != 'complete':
        raise ValueError('Training is incomplete; rerun the train command')
    path = model_dir/'lightgbm_model.txt'
    if sha256(path) != metadata['model_sha256']:
        raise ValueError('Model file does not match its recorded threshold/metadata')
    if sha256(root/'train'/'tfidf.npz') != metadata['signature']['tfidf_sha256']:
        raise ValueError('TF-IDF changed after training; do not mix feature/model runs')
    booster = lgb.Booster(model_file=str(path))
    if booster.feature_name() != metadata['features']:
        raise ValueError('Model feature names/order do not match metadata')
    return booster, metadata


def predict(root, split, model_dir=None, output=None):
    root = Path(root).resolve()
    booster, metadata = load_model(root, model_dir)
    work = root / ('train' if split == 'validation' else 'test')
    if feature_names(work) != metadata['features']:
        raise ValueError('Prediction features differ from training features/order')
    # Test preparation must use the frozen training TF-IDF, never a different work tree.
    manifest = work/'run_manifest.json'
    if manifest.exists() and split == 'test':
        sig = json.loads(manifest.read_text())['signature']
        fitted = sig['inputs'][-1]
        if fitted != file_identity(root/'train'/'tfidf.npz'):
            # Existing preparation fingerprint uses "bytes" rather than "size".
            expected = file_identity(root/'train'/'tfidf.npz')
            expected['bytes'] = expected.pop('size')
            if fitted != expected:
                raise ValueError('Test features were generated with a different TF-IDF artifact')
    output = Path(output) if output else work / f'{split}_probabilities.parquet'
    if output.suffix != '.parquet':
        raise ValueError('Prediction output must end in .parquet')
    cfg = metadata['signature']['config']
    count = _predict_file(booster, work/f'{split}_features.parquet', output, metadata['features'], cfg['batch_size'], cfg['num_threads'])
    dump_json(output.with_suffix('.metadata.json'), dict(model_sha256=metadata['model_sha256'],
        source=file_identity(work/f'{split}_features.parquet'), rows=count, threshold=metadata['decision_threshold']))
    print(f'Predicted {count:,} pairs: {output}', flush=True)
    return output, metadata
