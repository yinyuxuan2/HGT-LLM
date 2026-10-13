# Amazon five-domain Type-Aware Prompt HGT server training

The bundle already contains the frozen protocol, candidates, negatives,
Type-Aware BGE-M3 node features, and three-relation graph metadata. Raw Amazon
files and the BGE-M3 model are not required for training.

Recommended environment: Linux, Python 3.10+, NVIDIA driver compatible with
the selected PyTorch CUDA wheel, at least 16 GiB GPU memory (24 GiB preferred),
32 GiB system RAM, and 10 GiB free disk space per concurrently running seed.

Install PyTorch using the command from pytorch.org that matches the server
driver, then install the remaining dependencies:

```bash
python -m pip install -r requirements-server.txt
chmod +x run_seed.sh run_all_seeds.sh run_three_gpus.sh
```

One GPU, sequential and resumable:

```bash
nohup ./run_all_seeds.sh > logs/all-seeds-launcher.log 2>&1 &
```

Three GPUs, one seed per GPU:

```bash
./run_three_gpus.sh
```

Inspect progress:

```bash
tail -f logs/seed-42.log
nvidia-smi
```

Every completed epoch writes `last_checkpoint.pt`; rerunning the same command
resumes the seed. Final records include ranking metrics, AUC/AP, trainable
parameters, peak GPU memory, training time, and SHA-256 hashes.
