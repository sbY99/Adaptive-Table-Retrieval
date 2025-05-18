import os
import argparse
from dataloader import ATRDataset
from model import InferenceModule, load_model
from utils import read_json, read_jsonl, merge_dicts, set_seed
import logging
from accelerate import Accelerator

def str_to_boolean(str):
    if str == 't' or str == 'T' or str == 'True':
        return True
    elif str == 'f' or str == 'F' or str == 'False':
        return False
    else:
        raise ValueError('String must be t or T for True and f or F for False')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--test_file", type=str, default="data/spider_test_contriever.jsonl,data/bird_test_contriever.jsonl,data/spider2_contriever.jsonl")
    parser.add_argument("--test_join_data", type=str, default="data/test_join.json") 
    parser.add_argument("--ood_data_name", type=str, default="spider2")
    parser.add_argument("--k_list", type=str, default="")
    parser.add_argument("--is_train_thr", type=str_to_boolean, default=True)
    parser.add_argument("--output_file_path", type=str)
    parser.add_argument("--experiment_id", type=str, default="expr1")

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--sliding_window", type=int, default=20)
    parser.add_argument("--keep_table", type=int, default=10)
    parser.add_argument("--ood_sliding_window", type=int, default=10)
    parser.add_argument("--ood_keep_table", type=int, default=5)
    args = parser.parse_args()
    assert args.k_list == "" if args.is_train_thr else args.k_list.split(",")[0] != ""

    set_seed(42)

    logging.basicConfig(filename=f"{args.experiment_id}_evaluate.log", level=logging.INFO)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs("results", exist_ok=True)
    accelerator = Accelerator()
    trainer_id = args.model_name_or_path.split("/")[-1]

    with accelerator.main_process_first():
        model, tokenizer = load_model(model_dir=args.model_name_or_path)

    test_data_dict = {}
    test_dataset_dict = {}
    for test_file in args.test_file.split(","):
        key = test_file.split("/")[-1].replace(".jsonl","").replace("_test","")
        data = read_jsonl(test_file)
        test_data_dict[key] = data

    test_corpus_dict = merge_dicts(read_json("data/meta/spider-test-corpus-dict.json"), read_json("data/meta/bird-test-corpus-dict.json"))   
    test_corpus_dict = merge_dicts(test_corpus_dict, read_json("data/meta/spider2-corpus-dict.json"))
    for key, value in test_data_dict.items():
        if args.ood_data_name in key:
            test_dataset_dict[key] = ATRDataset(value, test_corpus_dict, tokenizer=tokenizer, max_length=args.max_length, join_data=[])
        else:
            test_dataset_dict[key] = ATRDataset(value, test_corpus_dict, tokenizer=tokenizer, max_length=args.max_length, join_data=[])

    inference_module = InferenceModule(
        trainer_id=trainer_id,
        accelerator=accelerator,
        model=model,
        test_dataset_dict=test_dataset_dict,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        sliding_window=args.sliding_window,
        keep_table=args.keep_table,
        ood_sliding_window=args.ood_sliding_window,
        ood_keep_table=args.ood_keep_table,
        ood_data_name=args.ood_data_name,
        k_list=[int(k) for k in args.k_list.split(",")] if not args.is_train_thr else [],
        input_file_path_list= [f for f in args.test_file.split(",")],
        output_file_path=args.output_file_path   
    )
    inference_module.inference()


if __name__ == "__main__":
    main()