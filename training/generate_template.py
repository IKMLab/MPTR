from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import torch
from tqdm import tqdm
import pickle
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def get_text(template, input_text_tuple, label, tokenizer):
    """
    This function is an adaptation of the `get_text` found in princeton-nlp's repository
    at https://github.com/princeton-nlp/LM-BFF/blob/main/tools/generate_template.py.
    """

    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    special_token_mapping = {
        "cls": tokenizer.cls_token_id,
        "mask": tokenizer.mask_token_id,
        "sep": tokenizer.sep_token_id,
        "sep+": tokenizer.sep_token_id,
    }
    for i in range(10):
        special_token_mapping["<extra_id_%d>" % (i)] = tokenizer._convert_token_to_id(
            "<extra_id_%d>" % (i)
        )
    template_list = template.split("*")
    input_ids = []
    for part in template_list:
        new_tokens = []
        if part in special_token_mapping:
            if part == "cls" and "T5" in type(tokenizer).__name__:
                # T5 does not have cls token
                continue
            new_tokens.append(special_token_mapping[part])
        elif part[:5] == "label":
            new_tokens += enc(" " + label)
        elif part[:6] == "sent_0":
            # sent_id = int(part.split("_")[1])
            new_tokens += enc(input_text_tuple)
        else:
            part = part.replace(
                "_", " "
            )  # there cannot be space in command, so use '_' to replace space
            # handle special case when t5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer._convert_token_to_id(part))
            else:
                new_tokens += enc(part)

        input_ids += new_tokens
    return input_ids


def generate(
    dataset,
    template,
    model,
    tokenizer,
    target_number,
    beam,
    label=None,
    length_limit=None,
    truncate=None,
):
    """
    This function is an adaptation of the `generate` found in princeton-nlp's repository
    at https://github.com/princeton-nlp/LM-BFF/blob/main/tools/generate_template.py.
    """
    input_tensors = []
    max_length = 0

    # Process the inputs
    for item in dataset:
        if label is None or item["label"] == label:
            input_text = get_text(
                template,
                item["text"],
                item["label"],
                tokenizer,
            )
            if truncate is not None:
                if truncate == "head":
                    input_text = input_text[-256:]
                elif truncate == "tail":
                    input_text = input_text[:256]
                else:
                    raise NotImplementedError
            input_ids = torch.tensor(input_text).long()
            max_length = max(max_length, input_ids.size(-1))
            input_tensors.append(input_ids)

    # Concatenate inputs as a batch
    input_ids = torch.zeros((len(input_tensors), max_length)).long()
    attention_mask = torch.zeros((len(input_tensors), max_length)).long()
    for i in range(len(input_tensors)):
        input_ids[i, : input_tensors[i].size(-1)] = input_tensors[i]
        attention_mask[i, : input_tensors[i].size(-1)] = 1

    # Print some examples
    print("####### example #######")
    print(tokenizer.decode(input_ids[0]))
    print(tokenizer.decode(input_ids[1]))
    print(tokenizer.decode(input_ids[2]))
    print("####### example #######\n")

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    assert len(input_tensors) > 0

    # Maximum generate content length
    max_length = 20

    start_mask = tokenizer._convert_token_to_id("<extra_id_0>")
    ori_decoder_input_ids = torch.zeros((input_ids.size(0), max_length)).long()
    ori_decoder_input_ids[..., 0] = model.config.decoder_start_token_id

    # decoder_input_ids: decoder inputs for next regressive generation
    # ll: log likelihood
    # output_id: which part of generated contents we are at
    # output: generated content so far
    # last_length (deprecated): how long we have generated for this part
    current_output = [
        {
            "decoder_input_ids": ori_decoder_input_ids,
            "ll": 0,
            "output_id": 1,
            "output": [],
            "last_length": -1,
        }
    ]
    for i in tqdm(range(max_length - 2)):
        new_current_output = []
        for item in current_output:
            # After the model generates <extra_id_1> 和 </s>, output_id will be 3,
            # which means the generation is done.
            if item["output_id"] > target_number:
                # Enough contents
                new_current_output.append(item)
                continue
            decoder_input_ids = item["decoder_input_ids"]

            # Forward
            batch_size = 32
            turn = input_ids.size(0) // batch_size
            if input_ids.size(0) % batch_size != 0:
                turn += 1
            aggr_output = []
            for t in range(turn):
                start = t * batch_size
                end = min((t + 1) * batch_size, input_ids.size(0))

                with torch.no_grad():
                    aggr_output.append(
                        model(
                            input_ids[start:end],
                            attention_mask=attention_mask[start:end],
                            decoder_input_ids=decoder_input_ids.cuda()[start:end],
                        )[0]
                    )
            aggr_output = torch.cat(aggr_output, 0)

            # Gather results across all input sentences, and sort generated tokens by log likelihood
            aggr_output = aggr_output.mean(0)
            log_denominator = torch.logsumexp(aggr_output[i], dim=-1).item()
            ids = list(range(model.config.vocab_size))
            ids.sort(key=lambda x: aggr_output[i][x].item(), reverse=True)
            ids = ids[: beam + 3]

            for word_id in ids:
                output_id = item["output_id"]

                if (
                    # start_mask is the token id of <extra_id_0>, id=32097
                    # <extra_id_1>: id = 32098
                    # This means that when the model generates <extra_id_1>, output_id will + 1
                    # When the model generates </s>, output_id will + 1
                    word_id == start_mask - output_id
                    or word_id == tokenizer._convert_token_to_id("</s>")
                ):
                    # Finish one part
                    if (
                        length_limit is not None
                        and item["last_length"] < length_limit[output_id - 1]
                    ):
                        check = False
                    else:
                        check = True
                    output_id += 1
                    last_length = 0
                else:
                    last_length = item["last_length"] + 1
                    check = True

                output_text = item["output"] + [word_id]
                # Perform softmax for log likelihood
                # negative sign of log_denominator means division
                ll = item["ll"] + aggr_output[i][word_id] - log_denominator
                new_decoder_input_ids = decoder_input_ids.new_zeros(
                    decoder_input_ids.size()
                )
                new_decoder_input_ids[:] = decoder_input_ids
                new_decoder_input_ids[..., i + 1] = word_id

                # Forbid single space token, "....", and ".........."
                if word_id in [3, 19794, 22354]:
                    check = False

                # Forbid continuous "."
                if (
                    len(output_text) > 1
                    and output_text[-2] == 5
                    and output_text[-1] == 5
                ):
                    check = False

                if check:
                    # Add new results to beam search pool
                    new_item = {
                        "decoder_input_ids": new_decoder_input_ids,
                        "ll": ll,
                        "output_id": output_id,
                        "output": output_text,
                        "last_length": last_length,
                    }
                    new_current_output.append(new_item)

        if len(new_current_output) == 0:
            break

        new_current_output.sort(key=lambda x: x["ll"], reverse=True)
        new_current_output = new_current_output[:beam]
        current_output = new_current_output

    result = []
    print("####### generated results #######")
    for item in current_output:
        generate_text = ""
        for token in item["output"]:
            generate_text += tokenizer._convert_id_to_token(token)
        print("--------------")
        print("score:", item["ll"].item())
        print("generated ids", item["output"])
        print("generated text", generate_text)
        result.append(generate_text)
    print("####### generated results #######\n")

    return result


def load_dataset(raw_train: pd.DataFrame, classes: dict) -> list:
    num_cols = raw_train.shape[1]
    if num_cols == 3:
        label_col = 2
        input_col = 1
    elif num_cols == 2:
        label_col = 1
        input_col = 0

    lines = raw_train.values.tolist()
    id_to_class = {i: c for c, i in classes.items()}

    dataset = []
    for line in lines:
        one_hot_idx = np.nonzero(line[label_col])[0]
        labels = [id_to_class[idx] for idx in one_hot_idx]
        tmp = [{"label": label, "text": line[input_col]} for label in labels]
        dataset.extend(tmp)

    return dataset


def search_template(
    model,
    tokenizer,
    dataset: list,
    beam: int,
    output_filename: str,
):
    f = open(output_filename, "w")

    # Single sentence tasks
    # We take two kinds of templates: put [MASK] at the beginning or the end
    template = "*cls**sent_0**<extra_id_0>**label**<extra_id_1>**sep+*"
    generate_text = generate(
        dataset,
        template,
        model,
        tokenizer,
        target_number=2,
        beam=beam,
        label=None,
        truncate="head",
    )[: beam // 2]

    print("####### generated templates #######")
    for text in generate_text:
        # Transform T5 outputs to our template format
        text = text.replace("<extra_id_0>", "*cls**sent_0*")
        text = text.replace("<extra_id_1>", "*mask*")
        # text = text.replace("<extra_id_2>", "*sep+*")
        text = text.replace("</s>", "*sep+*")
        text = text.replace("▁", "_")
        print(text)
        f.write(text + "\n")
    print("####### generated templates #######\n")

    template = "*cls*.*<extra_id_0>**label**<extra_id_1>**sent_0**sep+*"
    generate_text = generate(
        dataset,
        template,
        model,
        tokenizer,
        target_number=2,
        beam=beam,
        label=None,
        truncate="tail",
    )[: beam // 2]
    print("####### generated templates #######")
    for text in generate_text:
        # Transform T5 outputs to our template format
        text = text.replace("<extra_id_0>", "*cls*")
        text = text.replace("<extra_id_1>", "*mask*")
        text = text.replace("<extra_id_2>", "*+sent_0**sep+*")
        text = text.replace("</s>", "*+sent_0**sep+*")
        text = text.replace("▁", "_")
        print(text)
        f.write(text + "\n")
    print("####### generated templates #######\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t5_model",
        type=str,
        default="razent/SciFive-base-Pubmed_PMC",
        # default="t5-small",
        help="T5 pre-trained model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[42],
        # default=[42, 13, 21, 100, 87],
        help="Data split seeds",
    )
    parser.add_argument("--output_dir", type=str, default="auto_templates")
    parser.add_argument("--beam", type=int, default=100, help="Beam search width")
    parser.add_argument("--db_date", type=str, default="20230606_new", help="db_date")
    parser.add_argument("--num_labels", type=int, default=7)
    parser.add_argument("--data_type", default="train_all", type=str)

    args = parser.parse_args()
    # Build model-related variables
    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    tokenizer.sep_token = "</s>"
    output_dir = f"{args.output_dir}/{args.db_date}/{args.data_type}/{args.t5_model}"
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    model = model.cuda()
    model.eval()

    file = open(f"data/{args.db_date}/{args.num_labels}c/class_names.pkl", "rb")
    classes = pickle.load(file)

    data_dir = f"data/{args.db_date}/{args.num_labels}c/full"
    data_indice = sorted(
        [
            int(filename.stem.split("_")[-1])
            for filename in Path(f"{data_dir}/{args.data_type}").glob("train*.pkl")
        ]
    )
    # Manual label word mappings
    verbalizer = {
        "cyst": "cyst",
        "HCC": "hcc",  # hepatoma
        "cirrhosis": "cirrhosis",
        "post-treatment": "posttreatment",
        "steatosis": "steatosis",
        "metastasis": "metastasis",
        "hemangioma": "hemangioma",
    }
    new_classes = {verbalizer[c]: classes[c] for c in verbalizer.keys()}

    for data_index in data_indice:
        data = pd.read_pickle(f"{data_dir}/{args.data_type}/train_{data_index}.pkl")
        dataset = load_dataset(raw_train=data, classes=new_classes)

        search_template(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            beam=args.beam,
            output_filename=f"{output_dir}/{args.data_type}-{data_index}.txt",
        )


if __name__ == "__main__":
    main()
