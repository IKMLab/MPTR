from pathlib import Path
import torch
import os
import random
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from ast import literal_eval


def pred_by_threshold(
    t: float,
    y_true: np.array,
    similarities: np.array,
    classes: dict,
):
    preds = (similarities >= t) * 1
    sk_results = precision_recall_fscore_support(
        y_true,
        preds,
        # average="samples",  # For calculating sample-wise P and R scores.
    )
    outputs = {
        "f1": np.average(sk_results[2]),
        "P": np.average(sk_results[0]),
        "R": np.average(sk_results[1]),
    }
    for label_name, idx in classes.items():
        outputs[f"{label_name}_f1"] = sk_results[2][idx]
    return outputs


def get_avg_length(dataset: torch.utils.data.Dataset):
    all_lengths = 0
    data_size = len(dataset)
    for i in range(data_size):
        all_lengths += len(dataset[i]["input_ids"])
    return all_lengths / data_size


def load_csv_multi_label(filename: str, col_name: str = "labels") -> pd.DataFrame:
    """Prevent Pandas from converting lists of int into lists of strings.

    Args:
        filename (str): path of a csv file
        col_name (str, optional): column name of lists of int. Defaults to 'labels'.

    Returns:
        pd.DataFrame: a Pandas dataframe
    """
    return pd.read_csv(filename, converters={col_name: literal_eval})


def save_logged_results(filename: str, results: dict):
    try:
        old_df = pd.read_csv(filename)
        df = pd.concat([old_df, pd.DataFrame(results)], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame(results)

    df.to_csv(filename, index=None)


def set_seed(seed):
    """
    Args:
        seed: an integer number to initialize a pseudorandom number generator
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # if using more than one GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_baseline_table(
    y_preds: list,
    baseline_name: str,
    baseline_result_file: str = "results/baselines.pkl",
    all_doc_idx: list = None,
) -> None:
    if Path(baseline_result_file).exists():
        df = pd.read_pickle(baseline_result_file)
    else:
        assert all_doc_idx is not None
        df = pd.DataFrame({"doc_idx": all_doc_idx})

    df[baseline_name] = y_preds
    df.to_pickle(baseline_result_file)


def load_params(path_of_params):
    with open(path_of_params, "r") as f:
        params = json.load(f)
    return argparse.Namespace(**params)


def get_label_words(classes: list, use_multi_label_words=False) -> list:
    mapping = {
        "cyst": "cyst",
        "HCC": "hcc",  # hepatoma
        "cirrhosis": "cirrhosis",
        "post-treatment": "posttreatment",
        "steatosis": "steatosis",
        "metastasis": "metastasis",
        "hemangioma": "hemangioma",
    }
    if use_multi_label_words:
        mapping = {
            "cyst": ["cyst"],
            "HCC": ["hcc", "hepatoma"],  # hepatoma
            "cirrhosis": ["cirrhosis"],
            "post-treatment": ["posttreatment"],
            "steatosis": ["steatosis", "steatohepatitis"],
            "metastasis": ["metastasis"],
            "hemangioma": ["hemangioma"],
        }

    label_words = [mapping[c] for c in classes]
    return label_words


def seed_mapper(data_type: str) -> list:
    mapping = {"train_32": [0, 1, 3, 7, 10]}
    if data_type in mapping:
        return mapping[data_type]
    else:
        raise NotImplementedError
