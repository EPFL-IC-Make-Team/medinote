#!/bin/bash

python3 inference.py \
    --model_name meditron-7b-summarizer \
    --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-summarizer/hf_checkpoint/ \
    --data_path /pure-mlo-scratch/make_project/data/inference/meditron-7b-summarizer.jsonl \
    --output_path data/meditron-7b-summarizer.jsonl
