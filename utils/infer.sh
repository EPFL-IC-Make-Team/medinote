#!/bin/bash

# Accept an argument to ./infer.sh --model_name and redirect to the correct command
if [ "$1" == "meditron-7b-summarizer" || "$1" == "all" ]; then
    python3 utils/inference.py \
        --model_name meditron-7b-summarizer \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-summarizer/hf_checkpoint/ \
        --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
        --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
        --num_samples 10 \
        --mode summarizer \
        --template_path /pure-mlo-scratch/make_project/ClinicalNotes/generation/templates/template.json \
        --verbose
fi     
if [ "$1" == "meditron-7b-generator" || "$1" == "all" ]; then
    python3 utils/inference.py \
        --model_name meditron-7b-generator \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-generator/hf_checkpoint/ \
        --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
        --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
        --num_samples 1000 \
        --mode generator  \
        --verbose
fi
if [ "$1" == "meditron-7b-direct" || "$1" == "all" ]; then
    python3 utils/inference.py \
        --model_name meditron-7b-direct \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-generator/hf_checkpoint/ \
        --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
        --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
        --num_samples 1000 \
        --mode direct \
        --verbose
fi
if [ "$1" == "meditron-13b-summarizer" || "$1" == "all" ]; then
    python3 utils/inference.py \
        --model_name meditron-13b-summarizer \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-13b-summarizer/hf_checkpoint/ \
        --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
        --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
        --num_samples 1000 \
        --mode summarizer \
        --template_path /pure-mlo-scratch/make_project/ClinicalNotes/generation/templates/template.json \
        --verbose
fi
if [ "$1" == "meditron-13b-generator" || "$1" == "all" ]; then
    python3 utils/inference.py \
        --model_name meditron-13b-generator \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-13b-generator/hf_checkpoint/ \
        --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
        --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
        --num_samples 1000 \
        --mode generator \
        --verbose
fi
if [ "$1" == "meditron-13b-direct" || "$1" == "all" ]; then
    python3 utils/inference.py \
        --model_name meditron-13b-direct \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-13b-generator/hf_checkpoint/ \
        --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
        --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
        --num_samples 1000 \
        --mode direct \
        --verbose
fi