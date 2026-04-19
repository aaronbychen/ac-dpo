# Academic GPU Training Guide

This guide shows how to run AC-DPO training on the WashU Academic GPU cluster using the
`condo-cse5100` partition, without `tmux`. Training logs are written to `.out` files.

## 1. Log In

From your local machine:

```bash
ssh jiaxi.zhang@shell.engr.wustl.edu
```

Go to the repository:

```bash
cd /home/compute/jiaxi.zhang/workspace/ac-dpo
```

## 2. Request A GPU

Use the CSE5100 account:

```bash
srun -A engr-acad-cse5100 -p condo-cse5100 --gres=gpu:a40:1 --cpus-per-task=4 --mem=32G --time=08:00:00 --pty bash
```

If A40 is unavailable, try A6000:

```bash
srun -A engr-acad-cse5100 -p condo-cse5100 --gres=gpu:a6000:1 --cpus-per-task=4 --mem=32G --time=08:00:00 --pty bash
```

After the session starts, confirm that you are on a GPU node:

```bash
hostname
nvidia-smi
```

## 3. Activate The Conda Environment

Go back to the repository after the GPU session starts:

```bash
cd /home/compute/jiaxi.zhang/workspace/ac-dpo
```

Activate the project environment by full path:

```bash
source /home/compute/jiaxi.zhang/conda_envs/ac-dpo/bin/activate
```

Verify that the correct Python is active:

```bash
which python
```

It should point to:

```text
/home/compute/jiaxi.zhang/conda_envs/ac-dpo/bin/python
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 4. Set Environment Variables

Set Weights & Biases to offline mode so training does not wait for interactive input:

```bash
export WANDB_MODE=offline
```

Set Hugging Face cache locations:

```bash
export HF_HOME="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HOME/hf_cache/transformers"
export HF_DATASETS_CACHE="$HOME/hf_cache/datasets"
```

Optional Conda locations for future installs:

```bash
export CONDA_ENVS_DIRS="$HOME/conda_envs"
export CONDA_PKGS_DIRS="$HOME/conda_pkgs"
```

## 5. Run AC-DPO Training

From the repository root:

```bash
cd /home/compute/jiaxi.zhang/workspace/ac-dpo
```

Start training in the background and write logs to `acdpo.out`:

```bash
nohup python -m scripts.train_acdpo > acdpo.out 2>&1 &
```

Immediately check that the background job started:

```bash
jobs -l
```

Find the training process:

```bash
ps -u $USER -o pid,ppid,stat,etime,cmd | grep "scripts.train_acdpo"
```

Watch the log:

```bash
tail -f /home/compute/jiaxi.zhang/workspace/ac-dpo/acdpo.out
```

Press `Ctrl+C` to stop watching the log. This does not stop training.

To view recent log lines later:

```bash
tail -n 80 /home/compute/jiaxi.zhang/workspace/ac-dpo/acdpo.out
```

## 6. Monitor The Process

Find the training PID:

```bash
ps -u $USER -o pid,ppid,stat,etime,cmd | grep "scripts.train_acdpo"
```

Check whether a specific PID is still running:

```bash
ps -p <PID> -o pid,ppid,stat,etime,cmd
```

Monitor CPU and RAM for that PID:

```bash
watch -n 2 "ps -p <PID> -o pid,%cpu,%mem,rss,etime,cmd"
```

Monitor all GPUs:

```bash
watch -n 2 nvidia-smi
```

If you know the GPU index, monitor only that GPU. Use lowercase `-i`, not the number `-1`:

```bash
watch -n 2 nvidia-smi -i <GPU_INDEX>
```

Press `Ctrl+C` to stop `watch`.

## 7. Stop Training Manually

Find the training PID:

```bash
ps -u $USER -o pid,cmd | grep "scripts.train_acdpo"
```

Stop it:

```bash
kill <PID>
```

If it does not stop:

```bash
kill -9 <PID>
```

## 8. Run Baseline Training

Use the same GPU session, Conda environment, and environment variables.

From the repository root:

```bash
cd /home/compute/jiaxi.zhang/workspace/ac-dpo
```

Start baseline training in the background and write logs to `baseline.out`:

```bash
nohup python -m scripts.train_baseline > baseline.out 2>&1 &
```

Find the baseline process:

```bash
ps -u $USER -o pid,ppid,stat,etime,cmd | grep "scripts.train_baseline"
```

Watch the baseline log:

```bash
tail -f /home/compute/jiaxi.zhang/workspace/ac-dpo/baseline.out
```

Press `Ctrl+C` to stop watching the log. This does not stop training.

## Quick Command Block

Use this after the GPU session has started:

```bash
cd /home/compute/jiaxi.zhang/workspace/ac-dpo
source /home/compute/jiaxi.zhang/conda_envs/ac-dpo/bin/activate

export WANDB_MODE=offline
export HF_HOME="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HOME/hf_cache/transformers"
export HF_DATASETS_CACHE="$HOME/hf_cache/datasets"

nohup python -m scripts.train_acdpo > /home/compute/jiaxi.zhang/workspace/ac-dpo/acdpo.out 2>&1 &
jobs -l
ps -u $USER -o pid,ppid,stat,etime,cmd | grep "scripts.train_acdpo"
tail -f /home/compute/jiaxi.zhang/workspace/ac-dpo/acdpo.out
```
