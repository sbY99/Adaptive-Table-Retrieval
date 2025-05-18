import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import (
    AutoModel,
    AutoConfig,
    get_scheduler
)
from tqdm import tqdm
import torch.nn.functional as F
from accelerate import Accelerator
from sentence_transformers.losses import BatchHardTripletLossDistanceFunction

from dataloader import create_dataloader, make_sample_input
from utils import compute_prf1, merge_dicts, create_eval_step_list, log_to_file, read_jsonl, write_jsonl, write_json, \
    save_tmp_results, load_tmp_results, delete_path, get_anchor_negative_triplet_mask, get_anchor_positive_triplet_mask, get_key_by_value

class ATR(nn.Module):
    def __init__(self, model_name_or_path, tokenizer):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        thr_positions,
        tab_positions
    ):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        thr_positions: list of length= batch -> int index of [CLS]
        tab_positions: list of length= batch -> list of indices for [SEP]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        batch_size = input_ids.size(0)

        table_logits_list = []
        thr_logits_list = []

        table_embeds_list = []
        thr_embeds_list = []

        for b_idx in range(batch_size):
            thr_idx = thr_positions[b_idx]
            tab_idx_list = tab_positions[b_idx]

            # threshold logit
            thr_hidden = last_hidden_state[b_idx, thr_idx, :]   # (hidden_size,)
            thr_logit = self.classifier(thr_hidden)             # shape (1,)
            thr_embeds_list.append(thr_hidden)
            thr_logits_list.append(thr_logit)
            
            # table logits
            table_embeds_b = []
            table_logit_b = []
            for pos in tab_idx_list:
                tab_hidden = last_hidden_state[b_idx, pos, :]
                logit = self.classifier(tab_hidden)  # (1,)
                table_embeds_b.append(tab_hidden)
                table_logit_b.append(logit)

            assert len(table_logit_b) > 0
            table_embeds_b = torch.stack(table_embeds_b)  # (num_tables, hidden_size)
            table_logit_b = torch.cat(table_logit_b, dim=0)  # (num_tables, 1)

            table_embeds_list.append(table_embeds_b)
            table_logits_list.append(table_logit_b)


        return table_logits_list, thr_logits_list, table_embeds_list, thr_embeds_list


def adaptive_thresholding_loss(table_logits, thr_logit, table_embeds, thr_embed, pt_flags, group_ids, beta_l2=0.3, lambda_bce=0.3, gamma_cont=0.1, triplet_margin=5):
    """
    table_logits: (num_tables, 1)
    thr_logit: (1,)
    table_embeds: (num_tables, hidden_size)
    thr_embed: (1, hidden_size)
    pt_flags: list of 0 or 1, length = num_tables
    """
    # if table logits are truncated
    if len(table_logits) != len(pt_flags):
        pt_flags = pt_flags[:len(table_logits)]
    
    assert beta_l2 <= 1.0 and beta_l2 >= 0.0
    assert lambda_bce <= 1.0 and lambda_bce >= 0.0
    assert gamma_cont <= 1.0 and gamma_cont >= 0.0
    assert beta_l2 + lambda_bce + gamma_cont <= 1.0
    pt_indices = [i for i, f in enumerate(pt_flags) if f == 1]
    nt_indices = [i for i, f in enumerate(pt_flags) if f == 0]

    # L1: positive classes vs TH
    L1 = 0.0
    if len(pt_indices) > 0:
        pos_logits = table_logits[pt_indices]  # (num_pos,)
        pos_plus_thr = torch.cat([pos_logits, thr_logit], dim=0)  # (num_pos + 1,)

        sum_exp = torch.exp(pos_plus_thr).sum()
        loss_terms = []
        for i in range(len(pos_logits)):
            numerator = torch.exp(pos_logits[i])
            loss_terms.append(-torch.log(numerator / sum_exp))
        L1 = torch.stack(loss_terms).sum()

    # L2: negative classes + TH 
    L2 = 0.0
    if len(nt_indices) > 0:
        neg_logits = table_logits[nt_indices]
        neg_plus_thr = torch.cat([neg_logits, thr_logit], dim=0)

        sum_exp = torch.exp(neg_plus_thr).sum()
        thr_exp = torch.exp(thr_logit[0])
        L2 = -torch.log(thr_exp / sum_exp)
    
    # L3: BCE (table_logits vs pt_flags)
    L3 = 0.0
    target = torch.tensor(pt_flags, dtype=torch.float, device=table_logits.device)
    # reduction='sum' or 'mean'
    L3 = F.binary_cross_entropy_with_logits(table_logits, target, reduction='sum')

    # L4: Contrastive Loss
    L4 = 0.0
    eucledian_metric = BatchHardTripletLossDistanceFunction.eucledian_distance
    pairwise_dist = eucledian_metric(table_embeds)
    labels = torch.tensor(group_ids, device=table_embeds.device)

     # For each anchor, get the hardest positive
    mask_anchor_positive = get_anchor_positive_triplet_mask(labels).float()
    anchor_positive_dist = mask_anchor_positive * pairwise_dist
    hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True) # (num tables, 1)

    # For each anchor, get the hardest negative
    mask_anchor_negative = get_anchor_negative_triplet_mask(labels).float()
    max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True) # (num tables, 1)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    tl = hardest_positive_dist - hardest_negative_dist + triplet_margin
    tl[tl < 0] = 0
    L4 = tl.mean()

    return (1-beta_l2-lambda_bce-gamma_cont) * L1  + beta_l2 * L2 + lambda_bce * L3 + gamma_cont * L4


class ATRTrainer:
    def __init__(
        self,
        trainer_id,
        accelerator:Accelerator,
        model,
        train_dataset,
        valid_dataset,
        tokenizer,
        batch_size=2,
        lr=1e-5,
        epochs=3,
        scheduler_type="linear",
        max_length=2048,
        beta_l2=0.3,
        lambda_bce=0.3,
        gamma_cont=0.1,
        eval_num_per_epoch=2,
        sliding_window=0,
        keep_table=0,
        model_path=""
    ):
        self.trainer_id = trainer_id
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.beta_l2 = beta_l2
        self.lambda_bce = lambda_bce
        self.gamma_cont = gamma_cont
        self.eval_num_per_epoch = eval_num_per_epoch
        self.max_length = max_length

        self.prev_step = 0
        self.best_valid_loss = 9999.0
        self.best_perfect_recall = 0
        self.sliding_window = sliding_window
        self.keep_table = keep_table
        self.model_path = model_path

        train_dataloader = create_dataloader(
            train_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=max_length,
            shuffle=True,
            data_type="train"
        )
        valid_dataloader = create_dataloader(
            valid_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=max_length,
            shuffle=False,
            data_type="eval"
        )

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = get_scheduler(name=scheduler_type, optimizer=optimizer,
                                  num_warmup_steps=0, num_training_steps=len(train_dataloader)*epochs)

        self.model, self.optimizer, self.scheduler, self.train_dataloader, self.valid_dataloader = accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, valid_dataloader
        )


    def train(self):
        for epoch in range(1, self.epochs+1):
            self.model.train()
            total_loss = 0.0

            pbar = tqdm(enumerate(self.train_dataloader,start=1),
                        total=len(self.train_dataloader),
                        desc=f"Epoch {epoch}",
                        leave=True,
                        disable=not self.accelerator.is_local_main_process)
            total_steps = len(pbar)
            eval_step_list = create_eval_step_list(N=total_steps, M=self.eval_num_per_epoch)

            for step, batch in pbar:
                with self.accelerator.accumulate(self.model):
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    thr_positions = batch["thr_positions"]
                    tab_positions = batch["tab_positions"]
                    labels_list = batch["labels"]  # list of list(0/1)
                    group_ids_list = batch['group_ids_list']
                    overflow_gold_cnt_list = batch['overflow_gold_cnt_list']

                    table_logits_list, thr_logits_list, table_embeds_list, thr_embeds_list = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        thr_positions=thr_positions,
                        tab_positions=tab_positions
                    )

                    batch_loss = 0.0
                    b_size = input_ids.size(0)

                    for b_idx in range(b_size):
                        table_logits = table_logits_list[b_idx]
                        thr_logit = thr_logits_list[b_idx]
                        table_embeds = table_embeds_list[b_idx]
                        thr_embed = thr_embeds_list[b_idx]
                        pt_flags = labels_list[b_idx]
                        group_ids = group_ids_list[b_idx]

                        if table_logits.numel() == 0:
                            continue

                        loss_val = adaptive_thresholding_loss(table_logits, thr_logit, table_embeds, thr_embed, pt_flags, group_ids=group_ids, beta_l2=self.beta_l2, lambda_bce=self.lambda_bce, gamma_cont=self.gamma_cont)
                        batch_loss += loss_val

                    batch_loss = batch_loss / b_size
                    total_loss += batch_loss.item()

                    self.accelerator.backward(batch_loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                pbar.set_postfix({
                    "loss": f"{batch_loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.4g}"
                })

                if step in eval_step_list:
                    curr_step = self.prev_step + step * self.accelerator.num_processes
                    avg_train_loss = total_loss / step
                    if self.accelerator.is_main_process:
                        log_to_file(f"[  Epoch {epoch} | Step {curr_step}  ]")
                        log_to_file(f"  Train loss {avg_train_loss:.4f}")
                    self.accelerator.log({"Train loss":avg_train_loss}, step=curr_step)

                    # Validation
                    self.evaluate(epoch, curr_step)

            self.prev_step += step * self.accelerator.num_processes


    def evaluate(self, epoch, curr_step):
        self.model.eval()
        
        pred_list = []
        gold_sets = []
        total_loss = []

        pbar = tqdm(enumerate(self.valid_dataloader, start=1),
                    total=len(self.valid_dataloader),
                    desc="Validation",
                    leave=False,
                    disable=not self.accelerator.is_local_main_process)
        with torch.no_grad():
            for step, batch in pbar:
                query_ids = batch['query_id']
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                thr_positions = batch["thr_positions"]
                tab_positions = batch["tab_positions"]
                labels_list = batch["labels"]  # list of list(0/1)
                group_ids_list = batch['group_ids_list']
                overflow_gold_cnt_list = batch['overflow_gold_cnt_list']

                b_size = input_ids.size(0)
                for b_idx in range(b_size):
                    query_id = query_ids[b_idx]
                    input_id = input_ids[b_idx]
                    pt_flags = labels_list[b_idx]
                    group_ids = group_ids_list[b_idx]
                    overflow_gold_cnt = overflow_gold_cnt_list[b_idx]
                    finalized_thr_idx = None

                    # inference with sliding window approach
                    sample_text = self.tokenizer.decode([ii for ii in input_id if ii != self.tokenizer.pad_token_id])
                    query_text = sample_text.split("[SEP]")[0].replace("[CLS]","")
                    sample_table_list = sample_text.split("[SEP]")[1:-1]
                    tab_dict = {tab_idx:tab for tab_idx,tab in enumerate(sample_table_list)}
                    inversed_tab_keys = [i for i in range(len(tab_dict))][::-1] # we rerank the tables Nth to 1st

                    fixed_input_tab_idx_list = []
                    fixed_tab_cnt = self.sliding_window - self.keep_table
                    tail_idx=0
                    while tail_idx < len(inversed_tab_keys):
                        if tail_idx == 0:
                            fixed_input_tab_idx_list.append(inversed_tab_keys[:self.sliding_window]) # first input
                            tail_idx += self.sliding_window
                        else:
                            fixed_input_tab_idx_list.append(inversed_tab_keys[tail_idx:tail_idx+fixed_tab_cnt]) 
                            tail_idx += fixed_tab_cnt

                    loss_val = 0.0
                    pred_index_list = []
                    saved_input_tab_list = None # saved in the prev stage
                    for fixed_input_tab_idx in fixed_input_tab_idx_list: # e.g., 50, sliding 20, keep 10 -> [[49, 48, .. , 30], [29, .., 20], [19,..,10], [9,...,0]]
                        target_tab_idx_list = fixed_input_tab_idx[::-1] # reordering (0 to N)
                        target_tab_list = [tab_dict[t] for t in target_tab_idx_list]
                        
                        if len(target_tab_list) == self.sliding_window: # first input
                            pass
                        else:
                            target_tab_list += saved_input_tab_list # concat from saved tables / e.g., [20,..29] + [30, 32, 48, .., 34]
                            assert len(target_tab_list) == self.sliding_window, f"{len(target_tab_list)} {target_tab_list}"
                        
                        target_sample_input = make_sample_input(tokenizer=self.tokenizer, device=self.model.device, query_text=query_text, table_list=target_tab_list, window_size=self.sliding_window, max_token=self.max_length)
                        table_logits, thr_logit, table_embeds, thr_embed = self.model(**target_sample_input) # inference !
                        table_logits, thr_logit, table_embeds, thr_embed = table_logits[0], thr_logit[0], table_embeds[0], thr_embed[0] # batch size = 1
                        
                        if finalized_thr_idx is None:
                            thr_rank = (table_logits > thr_logit).sum().item()
                            if thr_rank >= self.keep_table: # threshold finalizing
                                finalized_thr_idx = len(tab_dict) - len(pred_index_list) - len(target_tab_idx_list) + thr_rank

                        _, top_indices = torch.topk(table_logits, k=self.keep_table)
                        saved_input_tab_list = [target_tab_list[t] for t in top_indices] # e.g., [30, 32, 48, .., 34]

                        # get finalized results
                        mask = torch.ones_like(table_logits, dtype=torch.bool)
                        mask[top_indices] = False
                        rest_values = table_logits[mask]
                        rest_indices = torch.arange(table_logits.size(0), device=table_logits.device)[mask]
                        rest_sorted_indices = torch.argsort(rest_values, descending=False)
                        rest_indices_sorted = rest_indices[rest_sorted_indices] # e.g., [39, .., 41]

                        # get real table idx / loop idx to real idx
                        for rest_indice in rest_indices_sorted:
                            rest_tab = target_tab_list[rest_indice]
                            real_tab_idx = get_key_by_value(tab_dict, rest_tab)
                            pred_index_list = [real_tab_idx] + pred_index_list # small to big

                        target_pt_flags = []
                        target_group_ids = []
                        for target_tab in target_tab_list: # keep the order
                            real_tab_idx = get_key_by_value(tab_dict, target_tab) 
                            target_pt_flags.append(pt_flags[real_tab_idx])
                            target_group_ids.append(group_ids[real_tab_idx])

                        loss_val += adaptive_thresholding_loss(table_logits, thr_logit, table_embeds, thr_embed, target_pt_flags, group_ids=target_group_ids,
                                                                beta_l2=self.beta_l2, lambda_bce=self.lambda_bce, gamma_cont=self.gamma_cont)
                    
                    loss_val /= len(fixed_input_tab_idx_list)
                    loss_val = self.accelerator.gather(loss_val)
                    try: # multi-gpu
                        total_loss += [b.item() for b in loss_val]
                    except:
                        total_loss.append(loss_val.item())
                    
                    # ignore the 'non-answerable' samples to calculate the metrics except for loss
                    if sum(pt_flags) == 0:
                        continue

                    if finalized_thr_idx is None:
                        finalized_thr_idx = thr_rank # if the threshold keep alive in the end, finalized thr idx is equal to the current rank
                    
                    for top_indice in top_indices.tolist()[::-1]: # reverse the order
                        top_tab = target_tab_list[top_indice]
                        real_tab_idx = get_key_by_value(tab_dict, top_tab)
                        pred_index_list = [real_tab_idx] + pred_index_list # small to big
                    assert len(pred_index_list) == len(list(set(pred_index_list)))

                    # if we don't train the thr value, we just rerank and truncate
                    if self.lambda_bce + self.gamma_cont == 1:
                        pred_index_list = pred_index_list[:10]
                    else:
                        pred_index_list = pred_index_list[:finalized_thr_idx]

                    gold_index_set = set(i for i, f in enumerate(pt_flags) if f == 1)
                    for i in range(1,overflow_gold_cnt+1):
                        gold_index_set.add(i*-1)
                    gold_sets.append((query_id, gold_index_set))
                    pred_list.append((query_id, pred_index_list))

                pbar.set_postfix({"val_step": step})

        save_tmp_results(data=pred_list, filename=f"tmp/{self.trainer_id}_pred{self.accelerator.local_process_index}.jsonl")
        save_tmp_results(data=gold_sets, filename=f"tmp/{self.trainer_id}_gold{self.accelerator.local_process_index}.jsonl")

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            query_id_list = [] # to avoid duplicated calculation
            pred_list = []
            gold_sets = []
            for process_idx in range(self.accelerator.num_processes):
                assert sum([p[0]!=g[0] for p, g in zip(load_tmp_results(f"tmp/{self.trainer_id}_pred{process_idx}.jsonl"), load_tmp_results(f"tmp/{self.trainer_id}_gold{process_idx}.jsonl"))]) == 0 # match the query ids
                pred_list += [d[1] for d in load_tmp_results(f"tmp/{self.trainer_id}_pred{process_idx}.jsonl") if d[0] not in query_id_list]
                gold_sets += [d[1] for d in load_tmp_results(f"tmp/{self.trainer_id}_gold{process_idx}.jsonl") if d[0] not in query_id_list]

                query_id_list += [d[0] for d in load_tmp_results(f"tmp/{self.trainer_id}_gold{process_idx}.jsonl") if d[0] not in query_id_list]

                delete_path(f"tmp/{self.trainer_id}_pred{process_idx}.jsonl")
                delete_path(f"tmp/{self.trainer_id}_gold{process_idx}.jsonl")

            avg_valid_loss = sum(total_loss)/len(total_loss)
            val_metrics = compute_prf1(preds=pred_list, labels=gold_sets)
            val_metrics = merge_dicts({f"Valid loss":avg_valid_loss}, val_metrics)
            val_metrics = {f"{k.capitalize()}":v for k,v in val_metrics.items()}
            for key, value in val_metrics.items():
                log_to_file(f"  {key} {value:.4f}")
            self.accelerator.log(val_metrics)
            log_to_file("")

            if avg_valid_loss < self.best_valid_loss:
                self.best_valid_loss = avg_valid_loss
                log_to_file(f"\n  Validation loss improved to {avg_valid_loss:.4f}")
                # Save the best model
                self.save_model("")
            else:
                log_to_file(f"\n  Validation loss did not improve: {avg_valid_loss:.4f} / Best: {self.best_valid_loss:.4f}")

            
    def save_model(self, postfix=""):
        """
        Save model and tokenizer to self.model_path.
        """
        model_path = self.model_path + postfix
        os.makedirs(model_path, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # 1) Save the base AutoModel
        unwrapped_model.model.save_pretrained(model_path)

        # 2) Save the classifier head
        classifier_path = os.path.join(model_path, "classifier.pt")
        torch.save(unwrapped_model.classifier.state_dict(), classifier_path)

        # 3) Save tokenizer
        unwrapped_model.tokenizer.save_pretrained(model_path)

        log_to_file(f"Model saved to {model_path}")


class InferenceModule:
    def __init__(self, trainer_id, accelerator:Accelerator, model, tokenizer, test_dataset_dict,
                 batch_size, max_length, sliding_window, keep_table, ood_sliding_window, ood_keep_table, ood_data_name, k_list, input_file_path_list, output_file_path):
        self.trainer_id = trainer_id
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.sliding_window = sliding_window
        self.keep_table = keep_table
        self.ood_sliding_window = ood_sliding_window
        self.ood_keep_table = ood_keep_table
        self.ood_data_name = ood_data_name
        self.k_list = k_list
        self.input_file_path_list = input_file_path_list
        self.output_file_path = output_file_path

        self.test_dataloader_dict = {}
        for i, (key, test_dataset) in enumerate(test_dataset_dict.items()):
            test_dataloader = create_dataloader(
                test_dataset,
                tokenizer=self.tokenizer,
                batch_size=self.batch_size,
                max_length=max_length,
                shuffle=False,
                data_type="eval"
            )
            if i ==0: # `accelerate.prepare()` requires you to pass at least one of training or evaluation dataloaders with `batch_size` attribute
                self.model, test_dataloader = accelerator.prepare(model, test_dataloader)
            else:
                test_dataloader = accelerator.prepare(test_dataloader)
            self.test_dataloader_dict[key.replace("_contriever","").replace("_uae","")] = test_dataloader # hard coding



    def inference(self):
        self.model.eval()
        for data_name, test_dataloader in self.test_dataloader_dict.items():
            if self.ood_data_name in data_name:
                sliding_window = self.ood_sliding_window
                keep_table = self.ood_keep_table
            else:
                sliding_window = self.sliding_window
                keep_table = self.keep_table

            pred_list = []
            gold_sets = []
            total_loss = []
            pbar = tqdm(enumerate(test_dataloader, start=1),
                        total=len(test_dataloader),
                        desc=f"Inference ({data_name}), Sliding Window ({sliding_window}), Keep Table ({keep_table})",
                        leave=False,
                        disable=not self.accelerator.is_local_main_process)
            
        
            with torch.no_grad():
                for step, batch in pbar:
                    query_ids = batch['query_id']
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    thr_positions = batch["thr_positions"]
                    tab_positions = batch["tab_positions"]
                    labels_list = batch["labels"]  # list of list(0/1)
                    group_ids_list = batch['group_ids_list']
                    overflow_gold_cnt_list = batch['overflow_gold_cnt_list']

                    b_size = input_ids.size(0)
                    for b_idx in range(b_size):
                        query_id = query_ids[b_idx]
                        input_id = input_ids[b_idx]
                        pt_flags = labels_list[b_idx]
                        group_ids = group_ids_list[b_idx]
                        overflow_gold_cnt = overflow_gold_cnt_list[b_idx]
                        finalized_thr_idx = None

                        # inference with sliding window approach
                        sample_text = self.tokenizer.decode([ii for ii in input_id if ii != self.tokenizer.pad_token_id])
                        query_text = sample_text.split("[SEP]")[0].replace("[CLS]","")
                        sample_table_list = sample_text.split("[SEP]")[1:-1]
                        tab_dict = {tab_idx:tab for tab_idx,tab in enumerate(sample_table_list)}
                        inversed_tab_keys = [i for i in range(len(tab_dict))][::-1] # we rerank the tables Nth to 1st

                        fixed_input_tab_idx_list = []
                        fixed_tab_cnt = sliding_window - keep_table
                        tail_idx=0
                        while tail_idx < len(inversed_tab_keys):
                            if tail_idx == 0:
                                fixed_input_tab_idx_list.append(inversed_tab_keys[:sliding_window]) # first input
                                tail_idx += sliding_window
                            else:
                                fixed_input_tab_idx_list.append(inversed_tab_keys[tail_idx:tail_idx+fixed_tab_cnt]) 
                                tail_idx += fixed_tab_cnt

                        pred_index_list = []
                        saved_input_tab_list = None # saved in the prev stage
                        for fixed_input_tab_idx in fixed_input_tab_idx_list: # e.g., 50, sliding 20, keep 10 -> [[49, 48, .. , 30], [29, .., 20], [19,..,10], [9,...,0]]
                            target_tab_idx_list = fixed_input_tab_idx[::-1] # reordering (0 to N)
                            target_tab_list = [tab_dict[t] for t in target_tab_idx_list]
                            
                            if len(target_tab_list) == sliding_window: # first input
                                pass
                            else:
                                target_tab_list += saved_input_tab_list # concat from saved tables / e.g., [20,..29] + [30, 32, 48, .., 34]
                                assert len(target_tab_list) == sliding_window, f"{len(target_tab_list)} {target_tab_list}"
                            
                            target_sample_input = make_sample_input(tokenizer=self.tokenizer, device=self.model.device, query_text=query_text, table_list=target_tab_list, window_size=sliding_window, max_token=self.max_length)
                            table_logits, thr_logit, table_embeds, thr_embed = self.model(**target_sample_input) # inference !
                            table_logits, thr_logit, table_embeds, thr_embed = table_logits[0], thr_logit[0], table_embeds[0], thr_embed[0] # batch size = 1
                            
                            if finalized_thr_idx is None:
                                thr_rank = (table_logits > thr_logit).sum().item()
                                if thr_rank >= keep_table: # threshold finalizing
                                    finalized_thr_idx = len(tab_dict) - len(pred_index_list) - len(target_tab_idx_list) + thr_rank

                            _, top_indices = torch.topk(table_logits, k=keep_table)
                            saved_input_tab_list = [target_tab_list[t] for t in top_indices] # e.g., [30, 32, 48, .., 34]

                            # get finalized results
                            mask = torch.ones_like(table_logits, dtype=torch.bool)
                            mask[top_indices] = False
                            rest_values = table_logits[mask]
                            rest_indices = torch.arange(table_logits.size(0), device=table_logits.device)[mask]
                            rest_sorted_indices = torch.argsort(rest_values, descending=False)
                            rest_indices_sorted = rest_indices[rest_sorted_indices] # e.g., [39, .., 41]

                            # get real table idx / loop idx to real idx
                            for rest_indice in rest_indices_sorted:
                                rest_tab = target_tab_list[rest_indice]
                                real_tab_idx = get_key_by_value(tab_dict, rest_tab)
                                pred_index_list = [real_tab_idx] + pred_index_list # small to big

                            target_pt_flags = []
                            target_group_ids = []
                            for target_tab in target_tab_list: # keep the order
                                real_tab_idx = get_key_by_value(tab_dict, target_tab) 
                                target_pt_flags.append(pt_flags[real_tab_idx])
                                target_group_ids.append(group_ids[real_tab_idx])

                        if finalized_thr_idx is None:
                            finalized_thr_idx = thr_rank # if the threshold keep alive in the end, finalized thr idx is equal to the current rank
                        
                        for top_indice in top_indices.tolist()[::-1]: # reverse the order
                            top_tab = target_tab_list[top_indice]
                            real_tab_idx = get_key_by_value(tab_dict, top_tab)
                            pred_index_list = [real_tab_idx] + pred_index_list # small to big
                        assert len(pred_index_list) == len(list(set(pred_index_list)))

                        # if we trained the thr class
                        if len(self.k_list) == 0:
                            pred_index_list = pred_index_list[:finalized_thr_idx]

                        gold_index_set = set(i for i, f in enumerate(pt_flags) if f == 1)        

                        for i in range(1,overflow_gold_cnt+1):
                            gold_index_set.add(i*-1)
                        gold_sets.append((query_id, gold_index_set))
                        pred_list.append((query_id, pred_index_list))

                    pbar.set_postfix({"inference_step": step})

            save_tmp_results(data=pred_list, filename=f"tmp/{self.trainer_id}_pred{self.accelerator.local_process_index}.jsonl")
            save_tmp_results(data=gold_sets, filename=f"tmp/{self.trainer_id}_gold{self.accelerator.local_process_index}.jsonl")

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                query_id_list = [] # to avoid duplicated calculation
                pred_list = []
                gold_sets = []

                for process_idx in range(self.accelerator.num_processes):
                    assert sum([p[0]!=g[0] for p, g in zip(load_tmp_results(f"tmp/{self.trainer_id}_pred{process_idx}.jsonl"), load_tmp_results(f"tmp/{self.trainer_id}_gold{process_idx}.jsonl"))]) == 0
                    pred_list += [d[1] for d in load_tmp_results(f"tmp/{self.trainer_id}_pred{process_idx}.jsonl") if d[0] not in query_id_list]
                    gold_sets += [d[1] for d in load_tmp_results(f"tmp/{self.trainer_id}_gold{process_idx}.jsonl") if d[0] not in query_id_list]

                    query_id_list += [d[0] for d in load_tmp_results(f"tmp/{self.trainer_id}_gold{process_idx}.jsonl") if d[0] not in query_id_list]

                    delete_path(f"tmp/{self.trainer_id}_pred{process_idx}.jsonl")
                    delete_path(f"tmp/{self.trainer_id}_gold{process_idx}.jsonl")

                
                # If we didn't train the threshold class
                # Metrics Logging (K = 2, 5, ...)
                if len(self.k_list) > 0 :
                    for eval_k in self.k_list:
                        val_metrics = compute_prf1(preds=[p[:eval_k] for p in pred_list], labels=gold_sets)
                        val_metrics = {f"[{data_name.capitalize()}] {k.capitalize()} @ {eval_k}: ":v for k,v in val_metrics.items()}
                        for key, value in val_metrics.items():
                            log_to_file(f"  {key} {value:.4f}")
                        self.accelerator.log(val_metrics)
                        log_to_file("")

                # If we trained the threshold class
                # there is no 'TopK'
                else:
                    val_metrics = compute_prf1(preds=pred_list, labels=gold_sets)
                    val_metrics = {f"[{data_name.capitalize()}] {k.capitalize()}: ":v for k,v in val_metrics.items()}
                    for key, value in val_metrics.items():
                        log_to_file(f"  {key} {value:.4f}")
                    self.accelerator.log(val_metrics)
                    log_to_file("")

                # Inference Result Logging
                for candidate_input_path in self.input_file_path_list:
                    if data_name.lower() in candidate_input_path.lower(): # choose current input file path
                        input_file_path = candidate_input_path
                        break
                inference_data = read_jsonl(input_file_path)
                output_file_path = self.output_file_path.replace(".jsonl",f"_{data_name}_window{sliding_window}_keep{keep_table}.jsonl")

                pred_dict = {}
                for q_id, pred in zip(query_id_list, pred_list):
                    pred_dict[q_id] = pred

                results = []
                for data in inference_data:
                    pred = pred_dict[data['query_id']] # keep order by query_id
                    db_tables = data["db_table_input"]
                    result_item = []
                    for p in pred:
                        result_item.append(db_tables[p]) # get [db, table]. Note that 'pred' is idx values.
                    results.append(result_item)
                write_jsonl(file_path=output_file_path, data=results)
                write_json(file_path=output_file_path.replace(".jsonl","_performance.json"), data=val_metrics)
                


def load_model(model_dir):
    config = AutoConfig.from_pretrained(model_dir)
    base_model = AutoModel.from_pretrained(model_dir, config=config)

    # We need a tokenizer for the constructor, so load it:
    # The line below loads whichever tokenizer files are in model_dir
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Create an empty model using the same architecture
    new_model = ATR(model_name_or_path=model_dir, tokenizer=tokenizer)
    new_model.model = base_model  # Replace the default base model with the loaded one

    # Load classifier weights
    classifier_path = os.path.join(model_dir, "classifier.pt")
    classifier_state = torch.load(classifier_path, weights_only=True)
    new_model.classifier.load_state_dict(classifier_state)

    new_model.eval()
    return new_model, tokenizer