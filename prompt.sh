#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 src/train.py --add_prompts=True --checkpoint='best_checkpoint.pt' --lr=3e-6