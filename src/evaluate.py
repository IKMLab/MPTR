import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from utils import seed_mapper


def get_f1_scores(y_true: list, y_pred: list):
    # All classes
    f1_scores = f1_score(y_true, y_pred, average=None)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")

    return f1_scores, macro_f1, micro_f1


def get_trained_model_f1s(
    class_dict: dict,
    seeds: list,
    data_type: str,
    method: str,
    y_true: list,
    union_metamap=None,
):
    macro_f1s = []
    micro_f1s = []
    for seed in seeds:
        preds = pd.read_pickle(
            f"results/{data_type}/{method}/seed_{seed}/predict/predictions.pkl"
        )
        y_pred = preds.y_pred.tolist()
        if union_metamap is not None:
            y_pred = do_union(y_pred, union_metamap)
        seed_result, macro_f1, micro_f1 = get_f1_scores(y_true, y_pred)
        macro_f1s.append(macro_f1)
        micro_f1s.append(micro_f1)
        if seed == 0:
            all_results = seed_result
        else:
            all_results = np.vstack((all_results, seed_result))
    avg_results = np.average(all_results, axis=0)
    std_results = np.std(all_results, axis=0)
    print("Average results:")
    for f1, label, std in zip(avg_results, class_dict, std_results):
        print(f"{label}: {f1:.3f} ({std:.3f})")
    print(f"Macro F1: {np.average(macro_f1s):.3f} ({np.std(macro_f1s):.3f})")
    print(f"Micro F1: {np.average(micro_f1s):.3f} ({np.std(micro_f1s):.3f})")


def do_union(base_preds: list, metamap_preds: list):
    union = []
    for base_pred, metamap_pred in zip(base_preds, metamap_preds):
        tmp = [x | y for x, y in zip(base_pred, metamap_pred)]
        union.append(tmp)
    return union


parser = argparse.ArgumentParser()
parser.add_argument("--data_type", type=str, default="train_32")
parser.add_argument("--test_filename", type=str, default="test_after_15_cleaned.pkl")
parser.add_argument(
    "--test_method",
    type=str,
    choices=["BERT", "MPTR", "BERT+MetaMap", "MPTR+AutoT"],
    default="MPTR",
)

args = parser.parse_args()
file = open(f"data/class_names.pkl", "rb")
classes = pickle.load(file)
test_df = pd.read_pickle(f"data/{args.test_filename}")
y_true = test_df.y_true.tolist()
seeds = seed_mapper(data_type=args.data_type)

if args.test_method == "BERT+MetaMap":
    metamap_preds = pd.read_pickle(f"results/metamap/predictions.pkl")
else:
    metamap_preds = None

get_trained_model_f1s(
    class_dict=classes,
    seeds=seeds,
    data_type=args.data_type,
    method=args.test_method,
    y_true=y_true,
    union_metamap=metamap_preds,
)
