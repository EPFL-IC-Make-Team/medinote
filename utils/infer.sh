#!/bin/bash

INPUT_PATH=/pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl
OUTPUT_PATH_7B=/pure-mlo-scratch/make_project/data/inference/generation_7B.jsonl
OUTPUT_PATH_13B=/pure-mlo-scratch/make_project/data/inference/generation_13B.jsonl
OUTPUT_PATH_GPT3=/pure-mlo-scratch/make_project/data/inference/generation_gpt3.jsonl
OUTPUT_PATH=/pure-mlo-scratch/make_project/data/evaluation/generation.jsonl
NUM_SAMPLES=3

if [ "$1" == "meditron-7b-summarizer" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-7b-summarizer \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-summarizer/hf_checkpoint/ \
        --input_path $INPUT_PATH \
        --output_path $OUTPUT_PATH_7B \
        --num_samples $NUM_SAMPLES \
        --mode summarizer \
        --template_path /pure-mlo-scratch/make_project/ClinicalNotes/generation/templates/template.json \
        --verbose
fi     
if [ "$1" == "meditron-7b-generator"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-7b-generator \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-generator/hf_checkpoint/ \
        --input_path $INPUT_PATH \
        --output_path $OUTPUT_PATH_7B \
        --num_samples $NUM_SAMPLES \
        --mode generator  \
        --verbose
fi
if [ "$1" == "meditron-7b-generator-gpt"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-7b-generator-gpt \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-generator/hf_checkpoint/ \
        --input_path $INPUT_PATH \
        --output_path $OUTPUT_PATH_7B \
        --num_samples $NUM_SAMPLES \
        --mode generator  \
        --verbose \
        --use_gpt_summary
fi
if [ "$1" == "meditron-7b-direct"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-7b-direct \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-direct/hf_checkpoint/ \
        --input_path $INPUT_PATH \
        --output_path $OUTPUT_PATH_7B \
        --num_samples $NUM_SAMPLES \
        --mode direct \
        --verbose
fi
if [ "$1" == "meditron-13b-summarizer"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-13b-summarizer \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-13b-summarizer/hf_checkpoint/ \
        --input_path $INPUT_PATH \
        --output_path $OUTPUT_PATH_13B \
        --num_samples $NUM_SAMPLES \
        --mode summarizer \
        --template_path /pure-mlo-scratch/make_project/ClinicalNotes/generation/templates/template.json \
        --verbose
fi
if [ "$1" == "meditron-13b-generator"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-13b-generator \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-13b-generator/hf_checkpoint/ \
        --input_path $INPUT_PATH \
        --output_path $OUTPUT_PATH_13B \
        --num_samples $NUM_SAMPLES \
        --mode generator \
        --verbose
fi
if [ "$1" == "meditron-13b-generator-gpt"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-13b-generator-gpt \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-13b-generator/hf_checkpoint/ \
        --input_path $INPUT_PATH \
        --output_path $OUTPUT_PATH_13B \
        --num_samples $NUM_SAMPLES \
        --mode generator  \
        --verbose
fi
if [ "$1" == "meditron-13b-direct"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-13b-direct \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-13b-direct/hf_checkpoint/ \
        --input_path $INPUT_PATH \
        --output_path $OUTPUT_PATH_13B \
        --num_samples $NUM_SAMPLES \
        --mode direct \
        --verbose
fi
if [ "$1" == "gpt3-direct" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name gpt-3.5-turbo \
        --input_path $INPUT_PATH \
        --output_path $OUTPUT_PATH_GPT3 \
        --train_path /pure-mlo-scratch/make_project/data/raw/summaries_full_train.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct-gpt \
        --verbose
fi

if [ "$1" == "combine" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --mode combine \
        --input_path $OUTPUT_PATH_7B,$OUTPUT_PATH_13B,$OUTPUT_PATH_GPT3 \
        --output_path $OUTPUT_PATH
fi