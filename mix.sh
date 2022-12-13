#!/bin/bash
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 python3 src/train.py --lr=1e-5 --batch_size=32  --n_epochs=20