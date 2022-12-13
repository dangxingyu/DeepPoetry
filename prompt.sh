#!/bin/bash
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 python3 src/train.py --lr=2e-6 --batch_size=32 
