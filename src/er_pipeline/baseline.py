"""Raw ZIP (or cleaned folder) -> trained baseline -> submission-format files."""
import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from .common import dump_json
from . import modeling, evaluation


def run(args):
    # Import here to avoid a CLI/baseline import cycle.
    from .cli import prepare, fingerprint, ensure_complete_test
    cfg = modeling.load_config(args.training_config, dict(
        num_boost_round=args.num_boost_round, num_threads=args.threads,
        max_train_pairs=args.max_train_pairs))
    root = args.work.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sample_test = None
    if args.input:
        source = args.input.resolve()
        if not source.exists():
            raise ValueError(f'Input not found: {source}')
        source_files = [source] if source.is_file() else sorted(source.rglob('*.tsv'))
        if not source_files:
            raise ValueError('Input directory contains no TSV files')
        identity = [fingerprint(p) for p in source_files]
        cleaner = Path(__file__).resolve().parents[2] / 'clean_er_data.py'
        signature = dict(inputs=identity, cleaner_sha256=modeling.sha256(cleaner))
        manifest_path = root / 'cleaning_manifest.json'
        cleaned = root / 'cleaned'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != signature:
                raise ValueError('Raw input/cleaner changed: choose a new --work folder')
            if not (cleaned/'cleaning_report.json').exists():
                raise ValueError('Completed cleaning output is missing: choose a new --work folder')
            print('Skip completed cleaning.', flush=True)
        else:
            subprocess.run([sys.executable, str(cleaner), '--input', str(source), '--output', str(cleaned)], check=True)
            dump_json(manifest_path, signature)
        if source.is_file():
            with zipfile.ZipFile(source) as archive:
                sample_test = any(Path(n).name == 'SAMPLE_README.txt' for n in archive.namelist())
        else:
            sample_test = any(source.rglob('SAMPLE_README.txt'))
    else:
        cleaned = args.cleaned.resolve()
    report_path = cleaned/'cleaning_report.json'
    if cfg['full_data'] and report_path.exists() and json.loads(report_path.read_text()).get('known_sample'):
        raise ValueError('Full-data training refuses a known sampled dataset')
    prepare(argparse.Namespace(cleaned=cleaned, work=root, split='train',
        stage='all', config=args.config, max_queries=None))
    metadata = modeling.train(root, cfg, args.model_dir)
    result = dict(validation_macro_f05=metadata['validation_macro_f05'],
        selected_threshold=metadata['decision_threshold'], trained=True,
        validation_note='Tuned holdout result; sampled reference pools yield optimistic estimates.')
    if not args.skip_test:
        test_cleaned = args.test_cleaned.resolve() if args.test_cleaned else cleaned
        prepare(argparse.Namespace(cleaned=test_cleaned, work=root, split='test',
            stage='all', config=args.config, max_queries=None))
        probabilities, metadata = modeling.predict(root, 'test', args.model_dir)
        ensure_complete_test(root)
        evaluation.submit(root/'test', probabilities, metadata['decision_threshold'])
        if args.test_cleaned:
            sample_test = None  # Cannot infer original-vs-sample provenance from folder name.
        result.update(matching_results=str(root/'test'/'matching_results.tsv'),
            candidate_pairs=str(root/'test'/'candidate_pairs.tsv'),
            test_known_sample=sample_test,
            submission_note='Use all original test sources and run the official validator. A sampled test set is not a valid challenge submission.')
        if sample_test:
            print('SAMPLE TEST OUTPUT: do not submit these files to the challenge; rerun test preparation using ALL original test records.', flush=True)
    dump_json(root/'baseline_result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result
