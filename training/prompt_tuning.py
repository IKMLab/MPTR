import os
import logging
import json
import pickle
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
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

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"logs/x_with_templates.log",
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%Y/%m/%d %H:%M",
    level=logging.INFO,
)
hf_logging.set_verbosity_info()


if __name__ == "__main__":
    args = parse_args()

    if args.k > 1 or args.t < 0.0:
        raise ValueError("The value k or t is wrongly defined!!")
    elif args.k == 0 and args.t == 0:
        raise ValueError("The value k and t should not be zeros.")
    elif args.k == 0:
        args.k = None
    elif args.t == 0:
        args.t = None

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

    if args.data_type.startswith("train_"):
        seed = args.seed
        data_indice = sorted(
            [
                int(filename.stem.split("_")[-1])
                for filename in Path(f"{data_dir}/{args.data_type}").glob("train*.pkl")
            ]
        )
        if len(data_indice) == 1 and args.data_type == "train_all":
            data_indice = data_indice * 5
            seeds = [0, 42, 100, 150, 385]
        for idx, data_index in enumerate(data_indice):
            if args.data_type == "train_all":
                set_seed(seeds[idx])
            else:
                seed = data_index
                set_seed(seed)

            train = pd.read_pickle(
                f"{data_dir}/{args.data_type}/train_{data_index}.pkl"
            )
            trainset = FewShotDataset(
                data=train,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
                template=args.template,
                prompt=args.prompt,
            )
            logger.info(f"Average training token length: {get_avg_length(trainset)}")
            if args.data_type == "full_valid":
                dev = pd.read_pickle(
                    f"{data_dir}/{args.data_type}/dev_{data_index}.pkl"
                )
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
            result_path = f"./results/{out_csv_name}/{exp_name}_idx{data_index}"
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

            if args.prompt != "":
                model = BertForPromptFinetuning.from_pretrained(
                    args.model_name,
                    use_multi_label_words=args.use_multi_label_words,
                )
                model.label_word_ids = label_word_ids
                p_tuning = True
            elif args.prompt == "":
                model = BertForSequenceClassification.from_pretrained(
                    args.model_name,
                    num_labels=len(label_words),
                    problem_type="multi_label_classification",
                )
                p_tuning = False

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
                    p_tuning=p_tuning,
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

                if not args.do_train:
                    exp_name = "zs_" + exp_name
                    break
            seed += 1
            save_args(args=args, save_dir=result_path)
            if not args.save_checkpoints:
                delete_large_files(result_path)
    else:
        raise ValueError("The data_type is wrongly defined!!")

    # log results
    results_log = {"exp_name": exp_name}
    results_log["template"] = args.template
    avg_precision = np.average(results["precision"])
    std_precision = np.std(results["precision"])
    avg_recall = np.average(results["recall"])
    std_recall = np.std(results["recall"])

    # log time
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    results_log["time"] = now

    # log all F1s of the classes in json
    results_log["all_f1s"] = json.dumps(F1s)

    # log averaged loss
    results_log["avg_loss"] = np.average(five_times_avg_loss)

    # results_log["all_avg_F1"] = np.average([v for vs in F1s.values() for v in vs])
    results_log["avg_F1"] = np.average(five_times_avg_f1)
    results_log["5_times_F1"] = str(five_times_avg_f1)
    stds = {k: np.std(vs) for k, vs in F1s.items()}
    results_log["all_avg_std"] = np.std(five_times_avg_f1)

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
        filename=f"results/{out_csv_name}_{args.num_labels}c.csv",
        results=results_log,
    )
    print(five_times_avg_f1)
