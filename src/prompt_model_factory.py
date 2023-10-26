from typing import Optional
from transformers import BertModel
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertOnlyMLMHead,
)
import torch


class BertForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config, use_multi_label_words: bool = False):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        # Initialize weights and apply final processing
        self.init_weights()

        self.label_word_ids = None
        self.use_multi_label_words = use_multi_label_words

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        mask_pos: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
        elif mask_pos is None:
            raise ValueError("`mask_pos` should be assigned!")

        # Encode everything
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # Get <mask> token representation
        sequence_output = outputs[0]
        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]

        # Logits over vocabulary tokens
        # prediction_mask_scores.shape: [batch_size, vocab_size]
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Return logits for each label
        logits = []
        if self.use_multi_label_words:
            for label_id in self.label_word_ids:
                one_label_logits = []
                # multiple ids in one label_id
                for id in label_id:
                    one_label_word_logits = prediction_mask_scores[:, id]
                    one_label_logits.append(one_label_word_logits.unsqueeze(-1))
                # one_label_logits: (bs, num_label_words)
                one_label_logits = torch.cat(one_label_logits, -1)
                # Get the max logits to choose the label word
                logits.append(torch.max(one_label_logits, dim=1, keepdim=True)[0])

        else:
            for label_id in range(len(self.label_word_ids)):
                logits.append(
                    prediction_mask_scores[:, self.label_word_ids[label_id]].unsqueeze(
                        -1
                    )
                )

        # logits.shape: [batch_size, num_classes]
        logits = torch.sigmoid(torch.cat(logits, -1))

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCELoss()
            loss = loss_fct(logits, labels.float())

        output = (logits, outputs.hidden_states) if output_hidden_states else (logits,)
        output = (output + (outputs.attentions)) if output_attentions else output

        return ((loss,) + output) if loss is not None else output
