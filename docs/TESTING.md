# Verification scope

Verified on CPU with a tiny locally constructed Transformer: actual LoRA cached
contrastive training and local resume; full cleaning-to-matching pipeline; entity
cluster separation; all-pair LightGBM training/reload; IVF-PQ and cosine reranking;
complete Source 1 coverage/singletons; memory rejection; embedding range allocation;
Training Job request construction; export excludes large training caches.

The 15 core tests produced 14 passes and one CUDA skip. The single-process hybrid
integration also passed. Python compilation and shell syntax checks passed.

The two-process integration was attempted: torchrun worker startup passed after
selecting loopback, but Gloo initialization failed with Operation not permitted
under this environment's networking restrictions. Gradient synchronization and
multi-process embedding integration are therefore NOT verified here. Run them on
a host that permits local distributed communication:

```bash
ER_TEST_DISTRIBUTED=1 python -m unittest discover -s tests -p test_hybrid.py -v
```

This development environment has no NVIDIA GPU or original full dataset. NCCL,
the CUDA container build, four-GPU encoder behavior, CUDA LightGBM, AWS permissions
and quotas, full-scale memory use, speed and F0.5 still require target verification.
The Training Job performs an actual four-rank NCCL all-reduce, a CUDA LightGBM fit,
and an encoder forward pass before preprocessing. LightGBM uses GPU 0 only.
