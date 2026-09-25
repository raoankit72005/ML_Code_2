# Reading progress

SageMaker CloudWatch shows stage names and JSON metrics. Training writes
encoder_rank0.jsonl through encoder_rank3.jsonl, embeddings_<split>_rank<r>.jsonl,
retrieval_<split>.jsonl, and lightgbm.jsonl under work/logs. TensorBoard events are
under work/logs/tensorboard. The job exports them into output.tar.gz.

Encoder loss is a sample-weighted average of rank-local cached contrastive losses.
Monitor recall is sampled and optimistic; compare final candidate_recall.json on
full references before judging retrieval improvements. LightGBM reports log loss
throughout boosting; full-query macro F0.5 and the selected threshold are logged
at the end. Training loss alone does not establish better entity matching.

Checkpoint files contain trainable adapters, optimizer state, per-rank PyTorch
RNG, epoch/cursor and signatures. Checkpoints are periodic, not every step.
A restart on an unchanged work volume replays only work after the last completed
checkpoint. Early-stage SQLite and feature outputs are needed as well; encoder
S3 checkpoints alone cannot resume a new Training Job.

The best initial/fine-tuned encoder is selected using the same fixed monitor.
The LightGBM training_snapshot.txt is diagnostic and is not automatically used
for exact continuation of a failed boosting fit.
