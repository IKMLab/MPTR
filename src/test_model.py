from prompt_model_factory import BertForPromptFinetuning
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)

# from prompt_tuning import compute_metrics
import torch
import pickle
import argparse
import pandas as pd
from prompt_dataset import FewShotDataset
from utils import load_params, get_label_words, seed_mapper, pred_by_threshold


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


parser = argparse.ArgumentParser()
parser.add_argument("--data_type", type=str, default="train_32")
parser.add_argument("--test_filename", type=str, default="test_after_15_cleaned.pkl")
parser.add_argument("--test_method", type=str, default="MPTR")
parser.add_argument("--ckpt_path", type=str, default="checkpoint-400")


infer_args = parser.parse_args()
method = "finetune" if infer_args.test_method == "BERT" else "prompt"
prompt_FT = True if "MPTR" in infer_args.test_method else False

test = pd.read_pickle(f"data/{infer_args.test_filename}")
if "cleaned" in infer_args.test_filename:
    use_cleaned_test_set = True

file = open(f"data/class_names.pkl", "rb")
classes = pickle.load(file)
class_names = list(classes.keys())

seeds = seed_mapper(data_type=infer_args.data_type)
for seed in seeds:
    device = (
        torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    )
    EXP_DIR = f"checkpoints/{infer_args.data_type}/{infer_args.test_method}/seed_{seed}"
    args = load_params(f"{EXP_DIR}/args.json")
    if prompt_FT:
        # To prevent mistakes
        assert args.prompt != ""

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_path = f"{EXP_DIR}/{infer_args.ckpt_path}"
    if prompt_FT:
        # Prompt tuning
        label_words = get_label_words(list(classes.keys()), args.use_multi_label_words)

        if args.use_multi_label_words:
            label_word_ids = []
            for l in label_words:
                one_label_ids = [tokenizer.convert_tokens_to_ids(word) for word in l]
                label_word_ids.append(one_label_ids)
        else:
            label_word_ids = (
                torch.tensor([tokenizer.convert_tokens_to_ids(l) for l in label_words])
                .long()
                .to(device)
            )
        model = BertForPromptFinetuning.from_pretrained(
            model_path,
            use_multi_label_words=args.use_multi_label_words,
        )
        model.label_word_ids = label_word_ids

    elif not prompt_FT:
        # Standard fine-tuning
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(class_names),
            problem_type="multi_label_classification",
        )

    result_path = (
        f"results/{infer_args.data_type}/{infer_args.test_method}/seed_{seed}/predict"
    )

    training_args = TrainingArguments(
        output_dir=result_path,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        evaluation_strategy="steps",
        logging_steps=100,  # same as eval_steps
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{args.best_metric}",
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics(
            threshold=args.t,
            classes=classes,
            p_tuning=prompt_FT,
        ),
    )

    testset = FewShotDataset(
        test,
        tokenizer,
        args.max_seq_len,
        template=args.template,
        prompt=args.prompt,
    )
    result = trainer.predict(testset)
    if not prompt_FT:
        pred_probs = torch.sigmoid(torch.tensor(result.predictions))
        preds = (pred_probs >= args.t) * 1
        test["y_pred"] = preds.numpy().tolist()

    elif prompt_FT:
        pred_probs = result.predictions
        preds = (pred_probs >= args.t) * 1
        test["y_pred"] = preds.tolist()

    with open(f"{result_path}/predictions.pkl", "wb") as f:
        pickle.dump(test, f)
