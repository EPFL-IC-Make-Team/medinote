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

if [ "$1" == "meditron-7b-generator-gold"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-7b-generator-gold \
        --model_path /pure-mlo-scratch/make_project/trial-runs/meditron-7b-generator/hf_checkpoint_new/ \
        --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
        --output_path /pure-mlo-scratch/make_project/data/inference/meditron-7b-generator-gold.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode generator-gold  \
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

if [ "$1" == "meditron-13b-generator-gold"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name meditron-13b-generator-gold \
        --model_path /pure-mlo-scratch/make_project/trial-runs/pubmed-13b-generator/hf_checkpoint_new/ \
        --input_path /pure-mlo-scratch/make_project/data/raw/summaries_full_test.jsonl \
        --output_path /pure-mlo-scratch/make_project/data/inference/meditron-13b-generator-gold.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode generator-gold \
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

# BASELINE: GPT 3.5 (from API)

# Dialogue -> Note (1-shot)
if [ "$1" == "gpt3-direct" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name gpt3-direct \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/gpt3-direct.jsonl \
        --train_path /pure-mlo-scratch/make_project/data/raw/summaries_full_train.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct-gpt \
        --shots 0 \
        --verbose $VERBOSE
fi
# GPT summary -> Note (0-shot)
if [ "$1" == "gpt3-generator-gpt" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name gpt3-generator-gpt \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/gpt3-generator-gpt.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode generator-gpt \
        --shots 0 \
        --verbose $VERBOSE
fi
# meditron-7b-summarizer's summary -> Note (0-shot)
if [ "$1" == "gpt3-generator-7b" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name gpt3-generator-7b \
        --input_path /pure-mlo-scratch/make_project/data/inference/meditron-7b-summarizer.jsonl \
        --output_path /pure-mlo-scratch/make_project/data/inference/gpt3-generator-7b.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode generator-gpt \
        --shots 0 \
        --verbose $VERBOSE
fi
# meditron-13b-summarizer's summary -> Note (0-shot)
if [ "$1" == "gpt3-generator-13b" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name gpt3-generator-13b \
        --input_path /pure-mlo-scratch/make_project/data/inference/meditron-13b-summarizer.jsonl \
        --output_path /pure-mlo-scratch/make_project/data/inference/gpt3-generator-13b.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode generator-gpt \
        --verbose $VERBOSE
fi

#Baseline GPT-4 (from API) (as reference)
if [ "$1" == "gpt4-direct" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name gpt4-direct \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/gpt4-direct.jsonl \
        --train_path /pure-mlo-scratch/make_project/data/raw/summaries_full_train.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct-gpt \
        --shots 0 \
        --verbose $VERBOSE
fi

# BASELINE: LLama-2 (7B/13B) (from local weights)
if [ "$1" == "llama-2-7b-direct"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name llama-2-7b-chat \
        --model_path /pure-mlo-scratch/llama2/llama-2-7b-chat-hf \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/llama-2-7b-direct.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct \
        --verbose $VERBOSE
fi
if [ "$1" == "llama-2-13b-direct"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name llama-2-13b-chat \
        --model_path /pure-mlo-scratch/llama2/llama-2-13b-chat-hf \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/llama-2-13b-direct.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct \
        --verbose $VERBOSE
fi

# BASELINE: Mistral-7B-Instruct-v0.1 (from HF)
if [ "$1" == "mistral-7b-direct"  ] || [  "$1" == "all" ]; then
    python3 utils/infer.py \
        --model_name mistral-7b \
        --model_path mistralai/Mistral-7B-Instruct-v0.1 \
        --input_path $INPUT_PATH \
        --output_path /pure-mlo-scratch/make_project/data/inference/mistral-7b-direct.jsonl \
        --num_samples $NUM_SAMPLES \
        --mode direct \
        --verbose $VERBOSE
fi

# Combine inference into a single file
if [ "$1" == "combine" ] || [ "$1" == "all" ]; then
    python3 utils/infer.py \
        --mode combine \
        --input_path $INFER_DIR \
        --output_path $OUTPUT_PATH
fi