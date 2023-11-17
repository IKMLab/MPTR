from typing import Dict
import torch
from transformers import EvalPrediction
from src.utils import pred_by_threshold


def compute_metrics(
    threshold=None,
    classes=None,
    p_tuning=False,
):
    def compute_metric_threshold(eval_pred: EvalPrediction):
        return pred_by_threshold(
            t=threshold,
            y_true=eval_pred.label_ids,
            similarities=eval_pred.predictions
            if p_tuning
            else torch.sigmoid(torch.tensor(eval_pred.predictions)),
            classes=classes,
        )

    return compute_metric_threshold


def append_multi_label_f1(metrics: Dict[str, float], classes: dict, F1s: list):
    avg_f1 = 0
    # Append F1 scores of all classes
    for i, class_name in enumerate(classes.keys()):
        f1 = metrics[f"test_{class_name}_f1"]
        F1s[class_name].append(f1)
        avg_f1 += f1
    return avg_f1 / len(classes)
