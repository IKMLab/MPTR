import os
import logging
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    logging as hf_logging,
)
from src.prompt_dataset import FewShotDataset
from src.prompt_model_factory import BertForPromptFinetuning
from src.utils import (
    get_avg_length,
    save_logged_results,
    set_seed,
    get_label_words,
    delete_large_files,
    save_args,
)
from params import parse_args
from metric_utils import compute_metrics, append_multi_label_f1


def get_best_row(
    seed: int,
    db_date: str,
    data_type: str,
    num_labels: int,
    exp_name: str = "",
) -> pd.DataFrame:
    csv_path = (
        f"results/autoT_seed{seed}_{db_date}_prompt_{data_type}_{num_labels}c.csv"
    )
    df = pd.read_csv(csv_path)
    if exp_name != "":
        df = df[df["exp_name"] == exp_name]
    best_idx = df["avg_F1"].argmax()
    best_row = df.iloc[best_idx]
    return best_row, df


logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"logs/x_with_templates.log",
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%Y/%m/%d %H:%M",
    level=logging.INFO,
)
hf_logging.set_verbosity_info()

args = parse_args()

if args.k > 1 or args.t < 0.0:
    raise ValueError("The value k or t is wrongly defined!!")
elif args.k == 0 and args.t == 0:
    raise ValueError("The value k and t should not be zeros.")
elif args.k == 0:
    args.k = None
elif args.t == 0:
    args.t = None

# 防呆
assert args.prompt == "auto"
assert args.data_type != "regular_cv"

logger.info(args.__dict__)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

####################
exp_type = "finetune" if args.prompt == "" else "prompt"
out_csv_name = f"{args.db_date}_{exp_type}_{args.data_type}"

exp_name = ""
if args.report_filter != "full":
    exp_name += f"{args.report_filter}_"
if args.k:
    exp_name = (
        f"{exp_name}at{args.k}_lr{args.lr}_warm{args.warmup_ratio}_"
        + f"bs{args.batch_size}_e{args.num_epochs}_"
    )
elif args.t:
    exp_name = (
        f"{exp_name}thres{args.t}_lr{args.lr}_warm{args.warmup_ratio}_"
        + f"bs{args.batch_size}_e{args.num_epochs}_"
    )
if args.use_multi_label_words:
    exp_name += "multi_"

if args.exp_tag != "":
    exp_name += f"{args.exp_tag}_"

exp_name += f"best{args.best_metric}_gpu{'_'.join(args.gpu_id.split(','))}"
if not args.do_train:
    exp_name = "zs_" + exp_name
####################

data_dir = f"data/{args.db_date}/{args.num_labels}c/{args.report_filter}"
file = open(f"data/{args.db_date}/{args.num_labels}c/class_names.pkl", "rb")
classes = pickle.load(file)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

results = {"precision": [], "recall": []}
F1s = defaultdict(list)
five_times_avg_f1 = []
five_times_avg_loss = []
save_checkpoints = False

if args.data_type.startswith("train_"):
    seed = args.seed
    set_seed(seed)
    if args.template == "best":
        save_checkpoints = True
        best_row, df = get_best_row(
            seed,
            db_date=args.db_date,
            data_type=args.data_type,
            num_labels=args.num_labels,
            exp_name=exp_name,
        )
        args.template = best_row["template"]

    train = pd.read_pickle(f"{data_dir}/{args.data_type}/train_{seed}.pkl")
    trainset = FewShotDataset(
        data=train,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        template=args.template,
        prompt=args.prompt,
    )
    logger.info(f"Average training token length: {get_avg_length(trainset)}")
    if args.data_type == "full_valid":
        dev = pd.read_pickle(f"{data_dir}/{args.data_type}/dev_{seed}.pkl")
        devset = FewShotDataset(
            data=dev,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            template=args.template,
            prompt=args.prompt,
        )
    else:
        devset = trainset

    train_bs = args.batch_size
    result_path = f"./results/autoT_{out_csv_name}/{exp_name}_idx{seed}"
    training_args = TrainingArguments(
        output_dir=result_path,
        learning_rate=args.lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=32,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        seed=seed,
        evaluation_strategy="steps",
        logging_steps=100,  # same as eval_steps
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{args.best_metric}",
    )

    model = BertForPromptFinetuning.from_pretrained(
        args.model_name,
        use_multi_label_words=args.use_multi_label_words,
    )
    model.label_word_ids = label_word_ids

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=devset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics(
            # k=args.k,
            threshold=args.t,
            classes=classes,
            p_tuning=True,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    if args.do_train:
        trainer.train()

    if args.do_predict:
        logger.info("*** Predict ***")
        test = pd.read_pickle(f"{data_dir}/test_after_15.pkl")
        testset = FewShotDataset(
            test,
            tokenizer,
            args.max_seq_len,
            template=args.template,
            prompt=args.prompt,
        )
        result = trainer.predict(testset)
        results["precision"].append(result.metrics["test_P"])
        results["recall"].append(result.metrics["test_R"])
        avg_f1 = append_multi_label_f1(result.metrics, classes, F1s)
        five_times_avg_f1.append(avg_f1)
        five_times_avg_loss.append(result.metrics["test_loss"])
        print(F1s)

        # Save predictions
        pred_probs = result.predictions
        preds = (pred_probs >= args.t) * 1
        test["y_pred"] = preds.tolist()
        with open(f"{result_path}/predictions.pkl", "wb") as f:
            pickle.dump(test, f)

    save_args(args=args, save_dir=result_path)
    if not save_checkpoints:
        delete_large_files(result_path)


results_log = {"exp_name": exp_name}
results_log["template"] = args.template
avg_precision = np.average(results["precision"])
std_precision = np.std(results["precision"])
avg_recall = np.average(results["recall"])
std_recall = np.std(results["recall"])

# log results
# log averaged loss
results_log["avg_loss"] = np.average(five_times_avg_loss)

# results_log["all_avg_F1"] = np.average([v for vs in F1s.values() for v in vs])
results_log["avg_F1"] = np.average(five_times_avg_f1)
stds = {k: np.std(vs) for k, vs in F1s.items()}
results_log["all_avg_std"] = np.average(list(stds.values()))

logger.info(f"***The following scores are averaged in {args.num_exps} times***")
logger.info(f"Pecision @ {args.k}: {avg_precision} ({std_precision})")
logger.info(f"Recall @ {args.k}: {avg_recall} ({std_recall})")
for k, v in F1s.items():
    score = np.average(v)
    results_log[f"{k}_f"] = [score]
    results_log[f"{k}_f_std"] = stds[k]
    print(f"{k}: {score}")

# log label_words
for k, v in classes.items():
    results_log[k] = str(label_words[v])

save_logged_results(
    filename=f"results/autoT_seed{args.seed}_{out_csv_name}_{args.num_labels}c.csv",
    results=results_log,
)

if args.prompt == "auto":
    template_dir = (
        f"auto_templates/{args.db_date}/{args.data_type}/razent/SciFive-base-Pubmed_PMC"
    )
    score_file = f"{template_dir}/{args.data_type}-{args.seed}-score.txt"
    file_exist = Path(score_file).exists()
    if not file_exist:
        with open(score_file, "w") as f:
            f.write(f"{args.template}\t{np.average(five_times_avg_f1)}")
            f.write("\n")
    else:
        with open(score_file, "a") as f:
            f.write(f"{args.template}\t{np.average(five_times_avg_f1)}")
            f.write("\n")
