#!/bin/bash

INFER_CONFIG="evaluate_config.yaml"
RERANKING_MODEL="contriever"
TEST_FILE="data/spider_test_contriever.jsonl,data/bird_test_contriever.jsonl,data/spider2_contriever.jsonl"

BATCH_SIZE=4
MAX_LENGTH=8192

SLIDING_WINDOW=20
KEEP_TABLE=15
OOD_SLIDING_WINDOW=10
OOD_KEEP_TABLE=5

experiment_id="expr1" # Your Experiments ID

MODEL_BEST_CHECKPOINT_DIR="models/${experiment_id}"
OUTPUT_FILE_PATH="results/${experiment_id}.jsonl"

mkdir -p "tmp"
mkdir -p "models"
mkdir -p "results"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file "$INFER_CONFIG" evaluate.py \
  --model_name_or_path "$MODEL_BEST_CHECKPOINT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --test_file "$TEST_FILE" \
  --max_length "$MAX_LENGTH" \
  --sliding_window "$SLIDING_WINDOW" \
  --keep_table "$KEEP_TABLE" \
  --ood_data_name "spider2" \
  --ood_sliding_window "$OOD_SLIDING_WINDOW" \
  --ood_keep_table "$OOD_KEEP_TABLE" \
  --output_file_path "$OUTPUT_FILE_PATH" \
  --k_list "" \
  --is_train_thr "True" \
  --experiment_id "$experiment_id"

wait
