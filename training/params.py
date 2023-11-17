import argparse
import pprint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        default="*cls**sent_0*_[PROMPT]*mask*.*sep+*",
        type=str,
    )
    parser.add_argument(
        "--prompt", default="The_report_is_related_to", choices=["", "auto"]
    )
    parser.add_argument(
        "--num_labels",
        default=7,
        type=int,
    )
    parser.add_argument(
        "--report_filter",
        default="full",
        choices=["full", "imp", "rnn", "bert"],
        type=str,
    )
    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--model_name",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        type=str,
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=str,
    )
    parser.add_argument(
        "--cls_mode",
        default="multi_label",
        choices=["multi_label", "multi_class"],
        type=str,
    )
    parser.add_argument(
        "--k",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--t",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--db_date",
        type=str,
        help="Path for the few-shot data.",
    )
    parser.add_argument(
        "--best_metric",
        default="loss",
        choices=["f1", "loss"],
        help="Validation metric for selecting the best model.",
        type=str,
    )
    parser.add_argument(
        "--exp_tag",
        default="",
        help="tag for experiment name",
        type=str,
    )
    parser.add_argument(
        "--num_exps",
        default="5",
        type=int,
        help="Number of repeated experiments.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
    )
    parser.add_argument(
        "--num_epochs",
        default="5",
        type=int,
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        default="42",
        type=int,
        help="Seed for controlling randomness.",
    )
    parser.add_argument(
        "--lr",
        default=2e-5,
        type=float,
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--data_type",
        default="train_32",
        description="This shows the number of examples that a model was trained from.",
        type=str,
    )
    parser.add_argument(
        "--save_conf_matrix",
        action="store_true",
    )
    parser.add_argument(
        "--use_multi_label_words",
        action="store_true",
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
    )
    args = parser.parse_args()

    print("==========Now using the following arguments: =========")
    pp = pprint.PrettyPrinter(indent=0)
    pp.pprint(vars(args))
    print("=====================================================")

    return args
