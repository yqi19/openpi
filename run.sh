#!/bin/bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pi05_rlbench --exp_name pytorch_rlbench