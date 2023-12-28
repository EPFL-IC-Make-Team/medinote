#!/bin/bash

# 7B summarizer
# python3 utils/inference.py \
#     --model_name meditron-7b-summarizer \
#     --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-summarizer/hf_checkpoint/ \
#     --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
#     --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
#     --num_samples 1000 \
#     --mode summarizer \
#     --template_path /pure-mlo-scratch/make_project/ClinicalNotes/generation/templates/template.json \

# 7B generator (from summarizer's summaries)
python3 utils/inference.py \
    --model_name meditron-7b-generator \
    --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-generator/hf_checkpoint/ \
    --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
    --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
    --num_samples 1000 \
    --mode generator 

# 7B generator (from GPT-4's summaries)
python3 utils/inference.py \
    --model_name meditron-7b-generator \
    --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-generator/hf_checkpoint/ \
    --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
    --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
    --num_samples 1000 \
    --mode generator \
    --use_gpt_summary

# 7B direct
python3 utils/inference.py \
    --model_name meditron-7b-direct \
    --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-generator/hf_checkpoint/ \
    --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
    --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
    --num_samples 1000 \
    --mode direct 

# 13B summarizer
python3 utils/inference.py \
    --model_name meditron-13b-summarizer \
    --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-13b-summarizer/hf_checkpoint/ \
    --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
    --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
    --num_samples 1000 \
    --mode summarizer \
    --template_path /pure-mlo-scratch/make_project/ClinicalNotes/generation/templates/template.json \

# 13B generator (from summarizer's summaries)
python3 utils/inference.py \
    --model_name meditron-13b-generator \
    --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-13b-generator/hf_checkpoint/ \
    --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
    --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
    --num_samples 1000 \
    --mode generator

# 13B generator (from GPT-4's summaries)
python3 utils/inference.py \
    --model_name meditron-13b-generator \
    --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-13b-generator/hf_checkpoint/ \
    --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
    --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
    --num_samples 1000 \
    --mode generator

# 13B direct
python3 utils/inference.py \
    --model_name meditron-13b-direct \
    --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-13b-generator/hf_checkpoint/ \
    --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
    --output_path /pure-mlo-scratch/make_project/data/inference/generation.jsonl \
    --num_samples 1000 \
    --mode direct
    

