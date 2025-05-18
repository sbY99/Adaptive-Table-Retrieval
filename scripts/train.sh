#!/bin/bash

TRAIN_CONFIG="train_config.yaml"

BASE_MODEL_PATH="answerdotai/ModernBERT-large"
RERANKING_MODEL="contriever"

TRAIN_FILE="data/train.jsonl"
VALID_FILE="data/valid.jsonl"

BATCH_SIZE=2 # batch size (2) x accumulation (16) x GPU nums (2) = actual batch size (64).
EPOCHS=3
LEARNING_RATE=3e-5 
MAX_LENGTH=8192
SLIDING_WINDOW=20
KEEP_TABLE=15
OOD_SLIDING_WINDOW=10
OOD_KEEP_TABLE=5

beta_l2=0.03
lambda_bce=0.13
gamma_cont=0.04

experiment_id="expr1"  # Your Experiments ID

echo ""
echo "==================================="
echo "Running TRAINING with:"
echo "  beta_l2     = ${beta_l2}"
echo "  lambda_bce  = ${lambda_bce}"
echo "  gamma_cont  = ${gamma_cont}"
echo "==================================="

MODEL_SAVE_DIR="models/${experiment_id}"

mkdir -p "tmp"
mkdir -p "models"
mkdir -p "results"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file "$TRAIN_CONFIG" train.py \
  --model_name_or_path "$BASE_MODEL_PATH" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --beta_l2 "$beta_l2" \
  --lambda_bce "$lambda_bce" \
  --gamma_cont "$gamma_cont" \
  --eval_num_per_epoch 4 \
  --train_file "$TRAIN_FILE" \
  --valid_file "$VALID_FILE" \
  --max_length "$MAX_LENGTH" \
  --sliding_window "$SLIDING_WINDOW" \
  --keep_table "$KEEP_TABLE" \
  --ood_data_name "spider2" \
  --ood_sliding_window "$OOD_SLIDING_WINDOW" \
  --ood_keep_table "$OOD_KEEP_TABLE" \
  --output_dir "$MODEL_SAVE_DIR" \
  --experiment_id "$experiment_id"

echo ""
echo "==================================="
echo "TRAINING completed."
echo "==================================="
