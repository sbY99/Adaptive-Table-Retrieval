import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from utils import log_to_file,group_lists

class ATRDataset(Dataset):
    def __init__(self, data_list, corpus_dict, tokenizer, max_length=2048, join_data={}):
        """
        data_list: list of { "query": str, "db_table_input": [[db, table], ...], "db_table_output": [[db, table], ...] }
        tokenizer: HF tokenizer
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.overed_items = []

        blocked_cnt = 0

        for item in data_list:
            query_id = item["query_id"]
            query_text = item["query"]
            db_tables = item["db_table_input"]       # [[db, tb], [db, tb], ...]
            labels = item["db_table_output"]            # [[db, tb], ...]
            join_ids =[]                          # [[id1,id2,...], [id1,id4,...], ...]
            solo_id = -1
            for db, tab in db_tables:
                try:
                    db_tab_ids = list(join_data[f"{db}[SEP]{tab}"].values())
                except: # if a table can't join with any tables
                    db_tab_ids = [solo_id]
                    solo_id -= 1
                join_ids.append(db_tab_ids)
            group_ids = group_lists(join_ids)

            # Positive indices
            label_set = set(tuple(x) for x in labels)
            pt_flags = []
            for dbt in db_tables:
                pt_flags.append(1 if tuple(dbt) in label_set else 0)
            overflow_gold_cnt = len([l for l in labels if l not in db_tables])

            text_segments = [query_text]
            for db, tb in db_tables:
                cols = " ".join(corpus_dict[f"{db}[SEP]{tb}"]['cols'])
                text_segments.append(f"{db} {tb} {cols}")
            final_text = "[SEP]".join(text_segments)

            encoded = tokenizer(
                final_text,
                # truncation=True,
                # max_length=max_length,
                padding=False,   
                return_offsets_mapping=False,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"][0]      
            attention_mask = encoded["attention_mask"][0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            thr_idx = None
            for i, tok in enumerate(tokens):
                if tok == "[CLS]":
                    thr_idx = i
                    break
            sep_indices = [i for i, tok in enumerate(tokens) if tok == "[SEP]"]
            tab_idx_list = sep_indices[:len(db_tables)]

            self.samples.append({
                "query_id":query_id,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "thr_idx": thr_idx,
                "tab_idx_list": tab_idx_list,
                "pt_flags": pt_flags,
                "group_ids":group_ids,
                "overflow_gold_cnt":overflow_gold_cnt
            })
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def table_filter_collate_fn(batch, pad_token_id, max_length=2048, data_type="train"):
    """
    batch: list of dict {
      "input_ids": tensor(seq_len),
      "attention_mask": tensor(seq_len),
      "thr_idx": int,
      "tab_idx_list": [int, int, ...],
      "pt_flags": [0/1, ...]
      "group_ids_list": [[int,..],..]
      "overflow_table_cnt_list": [int,...]
    }
    """
    max_seq_len = max(sample["input_ids"].size(0) for sample in batch)
    if data_type == "train": # in eval stage, we don't truncate because of sliding window
        max_seq_len = min(max_seq_len, max_length)

    query_ids = []
    padded_input_ids = []
    padded_attention_masks = []
    thr_positions = []
    tab_positions = []
    labels_list = []
    group_ids_list = []
    overflow_gold_cnt_list = []

    for sample in batch:
        query_id = sample['query_id']
        pt_flags = sample["pt_flags"]
        group_ids = sample["group_ids"]
        overflow_gold_cnt = sample['overflow_gold_cnt']
        seq_len = sample["input_ids"].size(0)
        # clip if seq_len > max_seq_len
        if seq_len > max_seq_len:
            input_ids = sample["input_ids"][:max_seq_len]
            attention_mask = sample["attention_mask"][:max_seq_len]
            # thr_idx, tab_idx_list 도 clip 반영
            thr_idx = sample["thr_idx"] if sample["thr_idx"] < max_seq_len else (max_seq_len-1)
            tab_idx_list = [x for x in sample["tab_idx_list"] if x < max_seq_len]

        else:
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]
            thr_idx = sample["thr_idx"]
            tab_idx_list = sample["tab_idx_list"]

        pad_len = max_seq_len - input_ids.size(0)

        # pad
        padded_i = torch.cat([
            input_ids,
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ], dim=0)
        padded_m = torch.cat([
            attention_mask,
            torch.zeros(pad_len, dtype=torch.long)
        ], dim=0)

        query_ids.append(query_id)
        padded_input_ids.append(padded_i)
        padded_attention_masks.append(padded_m)
        thr_positions.append(thr_idx)
        tab_positions.append(tab_idx_list)
        labels_list.append(pt_flags)
        group_ids_list.append(group_ids)
        overflow_gold_cnt_list.append(overflow_gold_cnt)

    padded_input_ids = torch.stack(padded_input_ids, dim=0)       # (batch, max_seq_len)
    padded_attention_masks = torch.stack(padded_attention_masks, dim=0)  # (batch, max_seq_len)

    return {
        "query_id":query_ids,
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_masks,
        "thr_positions": thr_positions,
        "tab_positions": tab_positions,
        "labels": labels_list,
        "group_ids_list":group_ids_list,
        "overflow_gold_cnt_list": overflow_gold_cnt_list
    }


def create_dataloader(dataset, tokenizer, batch_size=4, max_length=2048, shuffle=True, data_type='train'):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda x: table_filter_collate_fn(
            x,
            pad_token_id=tokenizer.pad_token_id,
            max_length=max_length,
            data_type=data_type
        ),
        prefetch_factor=2
    )


def make_sample_input(tokenizer, device, query_text:str, table_list:list, window_size:int, max_token:int):
    # Input
    # elements of table_list: {db} {table} {cols} format text.

    # Output
    # input_ids, attention_mask, thr_positions, tab_positions
    text_list = [query_text]
    for tab in table_list:
        text_list.append(tab)
    final_text = "[SEP]".join(text_list)

    encoded = tokenizer(
        final_text,
        padding=False,
        return_offsets_mapping=False,
        return_tensors="pt"
    )
    input_ids = encoded['input_ids'][0]
    attention_mask = encoded['attention_mask'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    assert len(input_ids) < max_token

    thr_idx = None
    for i, tok in enumerate(tokens):
        if tok == "[CLS]":
            thr_idx = i
            break
    sep_indices = [i for i, tok in enumerate(tokens) if tok == "[SEP]"]
    tab_idx_list = sep_indices[:window_size] # remove unnecessary (last) sep token

    # make single batch
    return {"input_ids":input_ids.reshape(1,-1).to(device), "attention_mask":attention_mask.reshape(1,-1).to(device), "thr_positions":[thr_idx], "tab_positions":[tab_idx_list]}