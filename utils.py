import json
import logging
from collections import defaultdict, deque

import os
import shutil
import torch
from torch import Tensor

import random
import numpy as np

def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    return data_list

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def merge_dicts(dict1, dict2):
    return {**dict1, **dict2}

def log_to_file(message):
    logging.info(message)


def save_tmp_results(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for element in data:
            query_id, element = element
            if isinstance(element, set):
                json_obj = {
                    "type": "set",
                    "query_id":query_id,
                    "value": list(element)
                }
            else:
                json_obj = {
                    "type": "list",
                    "query_id":query_id,
                    "value": element
                }
            f.write(json.dumps(json_obj, ensure_ascii=False))
            f.write('\n')

def load_tmp_results(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line)
            query_id = json_obj['query_id']
            if json_obj["type"] == "set":
                restored = set(json_obj["value"])
            else:
                restored = json_obj["value"]
            result.append((query_id, restored))
    return result


def delete_path(path):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        pass

def create_eval_step_list(N, M):
    """
    N: total steps
    M: num eval per 1 epoch
    """
    step = N // M  
    result = []
    N,M = int(N),int(M)
    for i in range(1, M + 1):
        val = i * step
        if val >= N:
            val = N - 1
        result.append(val)
    
    result[-1] = N
    
    return result


def compute_prf1(preds:list[list], labels:list[set]):
    assert len(preds) == len(labels)
    assert type(preds) is list and type(labels) is list and type(preds[0]) is list # keep order of preds
    metrics = {
        "recall":0,
        "perfect_recall":0,
    }
    cnt = 0

    for pred, label in zip(preds, labels):
        prd = set(pred)
        tgt = label

        tp = len(prd & tgt)
        fn = len(tgt - prd)

        _recall =  tp / (tp + fn) if tp + fn != 0 else 0
        metrics['recall'] +=  _recall
        metrics['perfect_recall'] += tgt.issubset(prd)
        cnt+=1
    
    for key, value in metrics.items():
        metrics[key] = round(value/cnt,3)

    return metrics


def group_lists(list_of_lists):
    element_to_indices = defaultdict(list)
    for i, sub in enumerate(list_of_lists):
        for element in sub:
            element_to_indices[element].append(i)
    
    visited = set()
    group_assignment = [-1] * len(list_of_lists)  
    group_id = 0

    for i in range(len(list_of_lists)):
        if i not in visited:
            queue = deque([i])
            visited.add(i)
            group_assignment[i] = group_id

            while queue:
                current = queue.popleft()
                for elem in list_of_lists[current]:
                    for neighbor in element_to_indices[elem]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            group_assignment[neighbor] = group_id
                            queue.append(neighbor)
            
            group_id += 1

    return group_assignment


def get_triplet_mask(labels: Tensor) -> Tensor:
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices

def get_anchor_positive_triplet_mask(labels: Tensor) -> Tensor:
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    return labels_equal & indices_not_equal

def get_anchor_negative_triplet_mask(labels: Tensor) -> Tensor:
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))


def get_key_by_value(d, target_value):
    for key, value in d.items():
        if value == target_value:
            return key
    return None 

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)