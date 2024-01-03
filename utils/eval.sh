#!/bin/bash

DATA_PATH=/pure-mlo-scratch/make_project/data/evaluation/generation.jsonl

if [ "$1" == "summary" ] || [ "$1" == "all" ]; then
    python3 utils/eval.py \
        --mode summary \
        --path $DATA_PATH \
        --score_types all
fi

if [ "$1" == "note" ] || [ "$1" == "all" ]; then
    python3 utils/eval.py \
        --mode note \
        --path $DATA_PATH \
        --score_types all
fi