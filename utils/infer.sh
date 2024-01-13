#!/bin/bash

INFER_DIR=/pure-mlo-scratch/make_project/data/inference/
INPUT_PATH=/pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl
OUTPUT_PATH=/pure-mlo-scratch/make_project/data/evaluation/generation.jsonl
NUM_SAMPLES=1000
VERBOSE=1

# Models: Meditron 7B, Meditron 13B
if [ "$1" == "meditron-7b-summarizer" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-7b-summarizer \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-summarizer/hf_checkpoint_new/ \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/meditron-7b-summarizer.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode summarizer \
        --template_path /pure-mlo-scratch/make_project/ClinicalNotes/generation/templates/template.json \
        --verbose $VERBOSE
fi     
if [ "$1" == "meditron-7b-generator"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-7b-generator \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-generator/hf_checkpoint_new/ \
        --input_path /pure-mlo-scratch/make_project/data/inference/meditron-7b-summarizer.jsonl \
        --output_path /pure-mlo-scratch/make_project/data/inference/meditron-7b-generator.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode generator  \
        --verbose $VERBOSE
fi
if [ "$1" == "meditron-7b-generator-gpt"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-7b-generator-gpt \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-generator/hf_checkpoint_new/ \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/meditron-7b-generator-gpt.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode generator-gpt  \
        --verbose $VERBOSE
fi
if [ "$1" == "meditron-7b-direct"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-7b-direct \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-direct/hf_checkpoint_new/ \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/meditron-7b-direct.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct \
        --verbose $VERBOSE
fi
if [ "$1" == "meditron-7b-direct-trunc"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-7b-direct-trunc \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-direct-trunc/hf_checkpoint_new \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/meditron-7b-direct-trunc.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct \
        --verbose $VERBOSE
fi
if [ "$1" == "meditron-13b-summarizer"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-13b-summarizer \
        --model_path /pure-mlo-scratch/make_project/trial-runs/pubmed-13b-summarizer/hf_checkpoint_new/ \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/meditron-13b-summarizer.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode summarizer \
        --template_path /pure-mlo-scratch/make_project/ClinicalNotes/generation/templates/template.json \
        --verbose $VERBOSE
fi
if [ "$1" == "meditron-13b-generator"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-13b-generator \
        --model_path /pure-mlo-scratch/make_project/trial-runs/pubmed-13b-generator/hf_checkpoint_new/ \
        --input_path /pure-mlo-scratch/make_project/data/inference/meditron-13b-summarizer.jsonl \
        --output_path /pure-mlo-scratch/make_project/data/inference/meditron-13b-generator.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode generator \
        --verbose $VERBOSE
fi
if [ "$1" == "meditron-13b-generator-gpt"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-13b-generator-gpt \
        --model_path /pure-mlo-scratch/make_project/trial-runs/pubmed-13b-generator/hf_checkpoint_new/ \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/meditron-13b-generator-gpt.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode generator-gpt  \
        --verbose $VERBOSE
fi
if [ "$1" == "meditron-13b-direct-trunc"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-13b-direct-trunc \
        --model_path /pure-mlo-scratch/make_project/trial-runs/pubmed-13b-direct-trunc/hf_checkpoint_new/ \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/meditron-13b-direct-trunc.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct \
        --verbose $VERBOSE
fi

# BASELINES: GPT 3.5, Llama 2 (7B & 13B), Mistral 7B
if [ "$1" == "gpt3-direct" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name gpt-3.5-turbo \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/gpt3-direct.jsonl \
        --train_path /pure-mlo-scratch/make_project/data/raw/summaries_full_train.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct-gpt \
        --verbose $VERBOSE
fi
if [ "$1" == "llama-2-7b-direct"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name llama-2-7b-chat-hf \
        --model_path /pure-mlo-scratch/llama2/llama-2-7b-chat \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/llama-2-7b-direct.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct \
        --verbose $VERBOSE
fi
if [ "$1" == "llama-2-13b-direct"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name llama-2-13b-chat-hf \
        --model_path /pure-mlo-scratch/llama2/llama-2-13b-chat \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/llama-2-13b-direct.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct \
        --verbose $VERBOSE
fi
if [ "$1" == "mistral-7b-direct"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name Mistral-7B-Instruct-v0.1 \
        --model_path mistralai/Mistral-7B-Instruct-v0.1 \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/Mistral-7b-direct.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct \
        --verbose $VERBOSE
fi
# PHI-2 not supported for now
# if [ "$1" == "phi-2"  ] || [  "$1" == "all" ]; then
#     python3 utils/infer.py \
#         --model_name phi-2 \
#         --model_path microsoft/phi-2 \
#         --input_path $INPUT_PATH \
#         --output_path /pure-mlo-scratch/make_project/data/inference/phi-2-direct.jsonl \
#         --num_samples $NUM_SAMPLES \
#         --mode direct \
#         --verbose $VERBOSE
# fi

# Combine inference into a single file
if [ "$1" == "combine" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --mode combine \
        --input_path $INFER_DIR \
        --output_path $OUTPUT_PATH
fi