# ML_Code_2: multi-GPU entity resolution on SageMaker Training Jobs

## One-GPU G5 baseline (experimental)

For `ml.g5.xlarge` (one A10G, 4 vCPUs, 16 GiB host RAM), build a **new** SM86 image:

```bash
CUDA_ARCH=86 bash sagemaker/build_push.sh
python sagemaker/launch.py --profile g5 \
  --role YOUR_ROLE_ARN --image YOUR_SM86_IMAGE_URI \
  --input s3://YOUR_BUCKET/ML_dataset.zip --output s3://YOUR_BUCKET/phase3-output
```

The launcher prints a dry run. Paid training starts only with `--submit`.
The G5 profile requests 200 GB of its fixed 250 GB local NVMe storage by
default; `--volume-gb` cannot expand it beyond 250 GB.
This profile uses one GPU for the LoRA encoder and embedding passes, two CPU
blocking workers, reduced retrieval budgets, and CPU LightGBM on at most
300,000 training and 50,000 early-stopping candidate pairs. All generated
candidate pairs are scored for final holdout and test outputs; the matcher
training sample and smaller retrieval budgets can lower recall/accuracy.
It still reads and indexes the full input dataset. With millions of rows,
CPU retrieval, disk usage, and end-to-end runtime on 4 vCPUs/16 GiB have **not**
been validated. A 24-hour maximum may stop the job before it produces a model.
Completed blocking chunks resume only if the entire work directory persists;
S3 encoder checkpoints alone cannot restart a new SageMaker job at blocking.
Benchmark on a representative subset before paying for a full-data run.

Streaming TSV cleaning and entity-cluster split → shared multilingual Siamese encoder → disk-backed embeddings → country/source IVF-PQ plus lexical blocking → merged candidates → string/address/retrieval/cosine features → CUDA LightGBM → full-validation macro F0.5 threshold → complete test matching results.

This is a separate project. It does not change ML_Code_1 or ML-code. The original dataset is used; no query or training-pair downsampling is enabled. A cluster-based holdout is reserved for validation. Retrieval shortlists and the final candidate cap are intentional approximations, not dataset sampling.

## Target hardware and actual device use

One SageMaker **ml.g6e.12xlarge**: 4 NVIDIA L40S GPUs (48 GB advertised per GPU), 48 vCPUs, 384 GiB host RAM. Runtime checks use detected available RAM/VRAM, including container memory limits. GPU memory is not one shared 192 GB allocation.

| Stage | Implementation |
|---|---|
| Cleaning, indexes, cluster split | Streaming CPU + SQLite; one writer; 120-second busy timeout |
| Encoder training | Four processes, identical shared-encoder replicas, synchronous sample-weighted LoRA gradient all-reduce via NCCL |
| Embeddings | Four GPUs process disjoint record-ID ranges; bounded batches, float16 disk shards |
| FAISS | CPU IVF-PQ, one country/source partition at a time; small partitions use exact search |
| Features | Streaming CPU computation; cosine reuses saved embeddings |
| Matcher | LightGBM 4.6.0 CUDA on GPU 0; all training pairs via disk-backed Sequence |
| Validation, output | Streamed full holdout and all original test S1 IDs |

Encoder negatives are local to each rank. The global batch is 512, normally 128 examples/rank; this is **not** a 512-negative contrastive denominator. Cached contrastive loss uses activation mini-batches of 16. True matches sharing an entity cluster never become negatives in one batch. Tiny tails are replicated across ranks to keep valid negative partners and equal step counts; no tail positives are dropped. LoRA updates query/value projections; the pretrained base is frozen. Initial pretrained-equivalent weights remain eligible for best-model selection if fine-tuning hurts monitor recall.

The multilingual checkpoint is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, pinned to revision `79f2382ceacceacdf38563d7c5d16b9ff8d725d6`. Only challenge-provided business records are used. Downloading the pretrained model requires outbound internet.

## Training Job workflow

Build the container once on a Linux x86_64 machine with Docker, AWS CLI, ECR permissions and enough disk. Do not run the old manual LightGBM source-checkout repair commands. The Dockerfile compiles **4.6.0** from its source distribution with CUDA enabled and SM89 selected, avoiding a CPU wheel or symlinked checkout. Compilation itself does not require a GPU.

```bash
bash sagemaker/build_push.sh
```

The script prints an ECR image URI. It creates `ml-code-2` in ECR if absent. Use your own SageMaker execution role ARN and an S3 output prefix; no credentials are stored in this repository. The role needs access to input/output/checkpoint S3 locations, the ECR image, and CloudWatch logs. The submitting identity needs CreateTrainingJob and PassRole permissions. Confirm the training-instance quota and availability in your region.

Install `boto3` in the **launcher environment** if missing:

```bash
python -m pip install boto3
```

Generate a reviewable request first (this does not start compute):

```bash
python sagemaker/launch.py \
  --role arn:aws:iam::YOUR_ACCOUNT:role/YOUR_SAGEMAKER_ROLE \
  --image YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ml-code-2:YOUR_TAG \
  --input s3://ml-hack1/ML_dataset.zip \
  --output s3://YOUR_BUCKET/ml-code-2 \
  --region us-east-1
```

Add `--submit` to that command to start paid training. Defaults: one `ml.g6e.12xlarge`, a requested 1000 GB work volume, a 24-hour maximum runtime, and On-Demand (not Spot). These are storage/runtime limits, **not** cost or completion-time estimates. On instances with local storage, SageMaker's actual volume provisioning rules may differ; the job checks actual free space. No job has been submitted by preparing this code.

SageMaker downloads the dataset channel under `/opt/ml/input/data/dataset`. The entry point accepts one ZIP or one original TSV dataset folder and runs `hybrid.py`. Input TSV delimiters are tabs, not commas. All generated intermediate files and model cache live under `/opt/ml/work`. Do not pass an S3 URI directly to `hybrid.py`; the launcher handles S3 input.

## Outputs and tracking

CloudWatch streams stage starts, encoder loss, processed pairs, throughput, sampled encoder recall, and LightGBM validation log loss. Each GPU rank also writes JSONL and TensorBoard events with process RSS and PyTorch GPU allocated/reserved/peak memory. Native FAISS/LightGBM allocations are not represented by PyTorch allocator metrics; use SageMaker hardware monitoring as well.

After success:

- SageMaker `model.tar.gz`: LoRA checkpoint, LightGBM model/threshold/features, TF-IDF state, configuration, source, and resolved environment.
- SageMaker `output.tar.gz`: **matching_results.tsv**, **candidate_pairs.tsv**, metrics/reports, JSONL/TensorBoard logs.
- Checkpoint S3 prefix: periodic encoder adapter/optimizer/RNG checkpoints.

Find artifact locations on the Training Job details page or with `aws sagemaker describe-training-job --training-job-name NAME`. Extract the output archive to obtain the TSVs. The model bundle is for reproducibility, not a SageMaker serving endpoint.

For a local run using the same work directory:

```bash
python hybrid.py --input /path/to/ML_dataset.zip --work /large-volume/work
python -m tensorboard.main --logdir /large-volume/work/logs/tensorboard --port 6006
```

Use a separate terminal for TensorBoard. Do not start a second pipeline on the same work directory; a process lock rejects it. Existing ML_Code_1 work directories are incompatible with this code/configuration.

## Memory and storage behavior

- Blocking uses 16 CPU processes by default. Each reads the shared SQLite index and writes an ordered 2,000-query chunk under `blocking_chunks/`. Completed chunks are reused when the same work volume is preserved; incomplete chunks restart. The final merge writes one ordered query TSV and candidate Parquet for downstream stages. Plan disk capacity for both the chunks and the merged output. The number of workers does not imply a linear speedup: benchmark query throughput on the target dataset before committing to a full paid run.
- 256 MiB SQLite caches per blocking worker, disk-backed tables, bounded Parquet groups and resource-aware LightGBM loading.
- One encoder replica per GPU; BF16 where supported, cached contrastive activation batches, synchronised OOM retries reducing activation size on all ranks. Tokenisation failures or irreducible OOM still fail with the last checkpoint retained.
- Embedding inference halves batches on CUDA OOM. Shards resume at committed row boundaries. Final merging uses a 16 MiB buffer and temporarily requires a second copy of the embedding data; shards are removed only after publication.
- 10 million 768-dimensional float16 embeddings consume 15.36 GB before indexes/features; feature tables and training staging can be much larger than raw data. Keep at least the configured 20 GiB disk reserve. Individual stages have additional disk guards.
- One CPU FAISS index is resident per country/source partition; an oversized partition fails its RAM estimate rather than allocating without a limit.
- LightGBM still needs resident bins and CUDA buffers. The preflight uses 65% of currently available host RAM and 75% of free GPU-0 memory. **Four encoder GPUs do not eliminate a single-GPU LightGBM memory limit.** If the conservative plan rejects the full pair table, the job stops with resource_plan.json. It never silently samples or pretends independent chunk training is equivalent to all-data boosting. Change candidate budgets only after measuring recall, or use a separately configured CPU matcher with the instance's larger RAM. The default remains CUDA.

There is no guarantee that every original full-data run fits or finishes in 24 hours. Estimate runtime from measured stage throughput. CPU cleaning, lexical retrieval and feature extraction are currently streamed serially; FAISS and LightGBM use configured CPU threads. GPU utilisation will vary by stage.

## Checkpoints and validation

Within a preserved work volume: completed stages skip, encoder resumes from last.pt, and embeddings resume by shard. Signatures reject changes in data/config/code. Index creation, retrieval, features and interrupted LightGBM fits may restart their incomplete stage. Full-pipeline state includes large SQLite/feature/vector files and is **not** uploaded as a checkpoint. A new Training Job does not automatically resume an interrupted job from just encoder S3 checkpoints. Use On-Demand; keep checkpoint prefixes unique per job. Full cross-job recovery needs the original work state and compatible signatures.

The encoder monitor uses a bounded sampled reference pool and is only for checkpoint selection. Final retrieval recall and threshold tuning use full reference pools and all holdout queries, including singletons and missed candidates. Macro F0.5 is computed per query with precision weight beta=0.5. Validation is reused for encoder selection, early stopping and threshold tuning; this is a tuned holdout result, not an unbiased test estimate. No improvement over the previous baseline is promised.

Final candidate_pairs.tsv contains exactly the candidates scored by the matcher, grouped by source1_entity_id. matching_results.tsv includes every original test Source 1 ID, with empty matched_entity_ids for abstentions. All matches must be among candidates. Final ID/order, duplicate, row-count and subset audits run automatically. Also run the challenge-supplied validator before leaderboard upload. This code repository is not a completed competition submission package until trained outputs and the requested methodology materials are added.

## Tests

```bash
python -m unittest discover -s tests -v
```

See docs/TESTING.md for verified and unverified scope. Configuration files are under configs/. Do not install notebook dependencies over a running job. Four-rank NCCL communication, CUDA LightGBM, and encoder smoke checks run at job start; CPU integration tests cannot establish NCCL/CUDA throughput or full-scale memory safety.
