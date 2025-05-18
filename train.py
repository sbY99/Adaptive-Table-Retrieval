from transformers import AutoTokenizer
import os
import argparse
from dataloader import ATRDataset
from model import ATR, ATRTrainer
from utils import read_json, read_jsonl, merge_dicts, set_seed
import logging
from accelerate import Accelerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="answerdotai/ModernBERT-large")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--beta_l2", type=float, default=0.3)
    parser.add_argument("--lambda_bce", type=float, default=0.3)
    parser.add_argument("--gamma_cont", type=float, default=0.1)
    parser.add_argument("--eval_num_per_epoch", type=int, default=2)

    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--valid_file", type=str, default="data/valid.jsonl")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--experiment_id", type=str, default="expr1")

    parser.add_argument("--train_valid_join_data", type=str, default="data/train_valid_join.json")
    parser.add_argument("--ood_data_name", type=str, default="spider2")

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--sliding_window", type=int, default=20)
    parser.add_argument("--keep_table", type=int, default=10)
    parser.add_argument("--ood_sliding_window", type=int, default=10)
    parser.add_argument("--ood_keep_table", type=int, default=5)
    args = parser.parse_args()

    set_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs("results", exist_ok=True)
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name="adaptive_table_reranker",config=args)
    
    backward_batch_size = args.batch_size * accelerator.num_processes * accelerator.deepspeed_plugin.gradient_accumulation_steps
    safe_model_name = args.model_name_or_path.split("/")[-1]

    trainer_id = f"{safe_model_name}_batch{backward_batch_size}_epoch{args.epochs}_lr{args.learning_rate}_beta{args.beta_l2}_lambda{args.lambda_bce}_gamma{args.gamma_cont}"
    
    if accelerator.is_main_process:
        accelerator.trackers[0].run.name = f"{safe_model_name}_batch{backward_batch_size}_epoch{args.epochs}_lr{args.learning_rate}_beta{args.beta_l2}_lambda{args.lambda_bce}_gamma{args.gamma_cont}"

    logging.basicConfig(filename=f"{args.experiment_id}_train.log", level=logging.INFO)

    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_data = read_jsonl(args.train_file)
    valid_data = read_jsonl(args.valid_file)


    train_valid_corpus_dict = merge_dicts(read_json("data/meta/spider-train-valid-corpus-dict.json"), read_json("data/meta/bird-train-valid-corpus-dict.json"))
    train_dataset = ATRDataset(train_data, train_valid_corpus_dict, tokenizer=tokenizer, max_length=args.max_length, join_data=read_json(args.train_valid_join_data))
    valid_dataset = ATRDataset(valid_data, train_valid_corpus_dict, tokenizer=tokenizer, max_length=args.max_length, join_data=read_json(args.train_valid_join_data))

    with accelerator.main_process_first():
        model = ATR(args.model_name_or_path, tokenizer)

    trainer = ATRTrainer(
        trainer_id=trainer_id,
        accelerator=accelerator,
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        epochs=args.epochs,
        scheduler_type=args.scheduler,
        beta_l2=args.beta_l2,
        lambda_bce=args.lambda_bce,
        gamma_cont=args.gamma_cont,
        max_length=args.max_length,
        eval_num_per_epoch=args.eval_num_per_epoch,
        sliding_window=args.sliding_window,
        keep_table=args.keep_table,
        model_path=args.output_dir
    )

    trainer.train()
    accelerator.end_training()    


if __name__ == "__main__":
    main()
