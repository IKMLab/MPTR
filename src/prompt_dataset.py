import torch
import pandas as pd


def get_prompt_length(tokenizer, prompt):
    return len(tokenizer.encode(prompt))


def tokenize_multipart_input(
    tokenizer,
    input_text_list: list,
    max_seq_len: int,
    template=None,
    prompt=None,
):
    """This function is an adaptation of the `tokenize_multipart_input` found in princeton-nlp's repository
    at https://github.com/princeton-nlp/LM-BFF/blob/main/src/dataset.py.

    Modifications include:
    - Extension of automatic prompt generation for multi-label classification.
    - Removal of parameters like `first_sent_limit`, `other_sent_limit`, `gpt3`, `truncate_head`, and `support_labels`.
    - Optimization of the code flow.

    Args:
        tokenizer: a pre-trained tokenizer from Hugging Face Transformers
        input_text_list (list): documents ready for tokenization.
        max_seq_len (int): max sequence length after adding the prompt along with special tokens from BERT.
        template (str, optional): placeholder for the prompt.
        prompt (str, optional): the prompt we use for input text.
    """
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    token_type_ids = []  # Only for BERT
    mask_pos = None  # Position of the mask token

    if prompt:
        special_token_mapping = {
            "cls": tokenizer.cls_token_id,
            "mask": tokenizer.mask_token_id,
            "sep": tokenizer.sep_token_id,
            "sep+": tokenizer.sep_token_id,
        }
        # Get variable list in the template
        if prompt != "auto":
            template = template.replace("[PROMPT]", prompt)
        template_list = template.split("*")
        if prompt == "auto":
            # find cls place
            cls_pos = template_list.index("cls")
            if template_list[cls_pos + 1] == "":
                # For these kinds of cases: *cls**sent_0*_Liver*mask*.*sep+*
                # Prompt is next to sent_0.
                prompt = template_list[cls_pos + 3]
            elif template_list[cls_pos + 1] != "" and (
                template_list[cls_pos + 1].startswith("_")
            ):
                # For these kinds of cases: *cls*_Liver*mask*.*+sent_0**sep+*
                # Prompt is next to cls.
                prompt = template_list[cls_pos + 1]
            if prompt.startswith("_"):
                prompt = prompt[1:]
        segment_id = 0

        for part in template_list:
            new_tokens = []
            segment_plus_1_flag = False
            if part in special_token_mapping:
                new_tokens.append(special_token_mapping[part])
                if part == "sep+":
                    segment_plus_1_flag = True
            elif part[:5] == "sent_" or part[:6] == "+sent_":
                sent_id = int(part.split("_")[1])
                max_len = max_seq_len - 3 - get_prompt_length(tokenizer, prompt)
                # Tokenize and truncate to max_seq_len
                tokens = enc(input_text_list[sent_id])[-max_len:]
                new_tokens += tokens
            else:
                # Just natural language prompt
                part = part.replace("_", " ")
                # handle special case when T5 tokenizer might add an extra space
                if len(part) == 1:
                    new_tokens.append(tokenizer.convert_tokens_to_ids(part))
                else:
                    new_tokens += enc(part)

            input_ids += new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]

            if segment_plus_1_flag:
                segment_id += 1

        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < max_seq_len

    else:
        input_ids = [tokenizer.cls_token_id]
        attention_mask = [1]
        token_type_ids = [0]
        max_len = max_seq_len - 2

        for sent_id, input_text in enumerate(input_text_list):
            if input_text is None:
                # Do not have text_b
                continue
            if pd.isna(input_text) or input_text is None:
                # Empty input
                input_text = ""
            input_tokens = enc(input_text)[:max_len] + [tokenizer.sep_token_id]
            input_ids += input_tokens
            attention_mask += [1 for i in range(len(input_tokens))]
            token_type_ids += [sent_id for i in range(len(input_tokens))]

    return input_ids, attention_mask, token_type_ids, mask_pos


class FewShotDataset(torch.utils.data.Dataset):
    """
    A class for creating the CGMH dataset in PyTorch.
    Currently, this class supports:
    (1) Few-shot data (e.g., train_size=16)
    (2) Small-size data (e.g., train_size>100)
    ---
    Attributes
        data (pd.DataFrame): the CGMH dataset
        tokenizer: a pre-trained HuggingFace tokenizer
        max_seq_len (int): maximum length for a sequence
        template (_type_, optional): template for the model. Defaults to None.
        prompt (_type_, optional): prompt for the model. Defaults to None.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_seq_len: int,
        template=None,
        prompt=None,
    ):
        self.template = template
        self.prompt = prompt
        self.docs = data.X.tolist()
        self.labels = data.y_true.tolist()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx):
        doc = self.docs[idx]
        input_ids, attn_mask, segs, mask_pos = tokenize_multipart_input(
            tokenizer=self.tokenizer,
            input_text_list=[doc],
            template=self.template,
            prompt=self.prompt,
            max_seq_len=self.max_seq_len,
        )
        item = {
            "input_ids": input_ids,
            "token_type_ids": segs,
            "attention_mask": attn_mask,
            "labels": [float(y) for y in self.labels[idx]],
        }
        if self.prompt:
            item["mask_pos"] = mask_pos
        return item

    def __len__(self):
        return len(self.docs)
