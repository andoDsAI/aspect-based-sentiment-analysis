import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel


class AspectModel(object):
    def __init__(self, config, args, aspect_label_list, polarity_label_lst):
        super(AspectModel, self).__init__()
        self.args = args
        self.num_aspects = len(aspect_label_list)
        self.num_polarities = len(polarity_label_lst)

        # Load pretrained
        if self.args.model_type == "phobert":
            self.roberta = RobertaModel(config)  # phobert
        else:
            self.roberta = XLMRobertaModel(config)  # XLM-Roberta

        self.dropout = nn.Dropout(args.dropout_rate)

        self.block_layers = []
        for _ in range(self.num_aspects):
            self.block_layers.append(
                nn.Linear(
                    4 * config.hidden_size,
                    args.hidden_size,
                )
            )

        self.aspect_classifier = nn.Linear(args.hidden_size, 1)
        self.polarity_classifier = nn.Linear(args.hidden_size, self.num_polarities)

        self.init_weights(self.polarity_classifier)

    def forward(
        self, input_ids, attention_mask, token_type_ids, aspect_label_ids, polarity_label_ids
    ):
        # hidden state
        bert_outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )[2]
        # concat 4 last outputs
        bert_outputs = torch.cat(
            (
                bert_outputs[-1][:, 0, ...],
                bert_outputs[-2][:, 0, ...],
                bert_outputs[-3][:, 0, ...],
                bert_outputs[-4][:, 0, ...],
            ),
            -1,
        )

        # get 6 output blocks
        aspect_outputs = []
        polarity_outputs = []
        for i in range(self.num_aspects):
            block_output = self.dropout(self.block_layers[i](bert_outputs))
            # aspect
            aspect_logits = self.dropout(self.aspect_classifier(block_output))
            aspect_logits = torch.sigmoid(aspect_logits)
            aspect_outputs.append(aspect_logits)
            # polarity
            polarity_logits = self.dropout(self.polarity_classifier(block_output))
            polarity_outputs.append(torch.softmax(polarity_logits))

        total_loss = 0
        if aspect_label_ids is not None:
            # Aspect loss
            aspect_loss_fct = nn.BCELoss()
            aspect_loss = aspect_loss_fct(aspect_outputs.view(-1), aspect_label_ids.view(-1))
            total_loss += self.args.aspect_loss_coef * aspect_loss

            # Polarity loss
            polarity_loss_fct = nn.CrossEntropyLoss()
            polarity_loss = polarity_loss_fct(
                polarity_outputs.view(-1, self.num_polarities),
                polarity_label_ids.view(-1, self.num_polarities),
            )

            total_loss += (1 - self.args.aspect_loss_coef) * polarity_loss

            return total_loss, aspect_outputs, polarity_outputs

        return aspect_outputs, polarity_outputs

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
