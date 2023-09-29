#!/bin/bash

python generate_huggingface.py --model /work/arjunguha-research-group/arjun/models/starcoderbase-1b\
    --prompt-file ./nate-dataset/sp14/pairs.jsonl \
    --output-file output-pairs.json \
    --batch-size 2
