# Retrieve Only Relevant Tables Whether Few or Many: Adaptive Table Retrieval Method

This repository contains the official implementation for the paper: *Retrieve Only Relevant Tables Whether Few or Many: Adaptive Table Retrieval Method*.

## Requirements

Set up your environment using:

```bash
conda env create --file environment.yaml
conda activate atr
```

## Hardware Requirements

### Training

* **Recommended Setup (Used in Paper)**: 2 × NVIDIA RTX A6000 GPUs, each with 48GB memory.

### Evaluation

* Single NVIDIA RTX A6000 or A100 GPU
* Evaluation is also possible on GPUs with less memory by reducing the batch size accordingly.

## Data

Download our preprocessed data here:
- [Google Drive Data Download Link](https://drive.google.com/file/d/1KSjM8RDsDCjzDZ2NEpnuvUr8_dmg4G5I/view?usp=sharing)

## Model Weights

You can download model weights here:
- [Google Drive Model Weights Download Link](https://drive.google.com/file/d/1x3iIhdrHRK7ku6XIFOfHEZwMz1B-4CBP/view?usp=sharing)


## Project Structure
Then, the project structure will be as follows:

```
ATR/
├── data/
│   ├── label/                        # Labels for test datasets
│   ├── meta/                         # Metadata for each dataset (table corpus information)
│   ├── train.jsonl
│   ├── valid.jsonl
│   ├── spider_test_contriever.jsonl  # Top-50 tables from Contriever (for re-ranking)
│   ├── bird_test_contriever.jsonl    # Top-50 tables from Contriever (for re-ranking)
│   └── spider2_contriever.jsonl      # Top-50 tables from Contriever (for re-ranking)
├── results/
│   ├── contriever/                   # ATR re-ranking results of top-50 tables retrieved by Contriever
│   ├── uae/                          # ATR re-ranking results of top-50 tables retrieved by UAE
│   └── results.ipynb                 # Code for calculating retrieval metrics
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
├── train.py
├── evaluate.py
├── model.py
├── dataloader.py
├── utils.py
├── train_config.yaml
├── evaluate_config.yaml
├── environment.yaml
└── README.md
```

## Training

Training uses the Accelerate framework with DeepSpeed. Configure training settings in `train_config.yaml` (provided in the repository):

* `distributed_type`: DEEPSPEED
* `zero_stage`: 2
* `gradient_accumulation_steps`: 16
* `gradient_clipping`: 1.0
* `num_processes`: 2 (GPUs used)

Key training parameters in `scripts/train.sh`:

* `batch_size`: 2 (`batch_size` 2 × `num_processes` 2 × `gradient_accumulation_steps` 16 = actual batch size 64)
* `epochs`: 3
* `learning_rate`: 3e-5
* `max_length`: 8192
* Adaptive thresholding parameters: `beta_l2` = 0.03, `lambda_bce` = 0.13, `gamma_cont` = 0.04
* Sliding window settings: Spider/BIRD (`sliding_window`: 20, `keep_table`: 15), Spider 2.0 (`ood_sliding_window`: 10, `ood_keep_table`: 5)

Run training with:

```bash
bash scripts/train.sh
```


## Evaluation

Evaluation is configured via `evaluate_config.yaml`:

* `distributed_type`: MULTI\_GPU
* `num_processes`: 2

Essential parameters for evaluation in `scripts/evaluate.sh`:

* `batch_size`: 4
* `max_length`: 8192
* Sliding window settings: Spider/BIRD (`sliding_window`: 20, `keep_table`: 15), Spider 2.0 (`ood_sliding_window`: 10, `ood_keep_table`: 5)

Run evaluation with:

```bash
bash scripts/evaluate.sh
```

Results and logs will be saved in the `./results` and `{experiment_id}.log` respectively.


## Model Performance

ATR achieves state-of-the-art results across multiple benchmarks:

| Model          | Spider R | Spider CR | BIRD R | BIRD CR | Spider 2.0 R | Spider 2.0 CR |
| -------------- | -------- | --------- | ------ | ------- | ---------- | ---------- |
| **ATR (Cont.)** | **99.5** | **99.2** | **98.2** | **96.0** | **72.4** | **64.4** |
| **ATR (UAE)** | **99.6** | **99.4** | **98.6** | **97.1** | **75.4** | **68.7** |

*Note:* The reported values may not be perfectly reproducible and might vary slightly due to factors such as the accelerate library, CUDA versions, specific GPU devices used, and other stochastic elements in the evaluation process.