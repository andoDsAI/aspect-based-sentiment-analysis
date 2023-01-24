import os

from tensorflow.keras import layers, models
from transformers import TFRobertaModel

from .modules import TransformerBlock


class AspectBasedModel:
    """Aspect-based sentiment analysis model."""

    def __init__(self, config, args, embedding_matrix, num_aspects, num_polarities, **kwargs):
        super(AspectBasedModel, self).__init__(**kwargs)
        self.args = args
        self.embedding_matrix = embedding_matrix
        self.num_aspects = num_aspects
        self.num_polarities = num_polarities

        self.input_ids = layers.Input(shape=(args.max_seq_len,), name="input_ids", dtype="int32")
        self.attention_mask = layers.Input(
            shape=(args.max_seq_len,), name="attention_mask", dtype="int32"
        )
        self.transformer_mask = layers.Input(
            shape=(args.max_seq_len, args.max_seq_len), name="transformer_mask", dtype="int32"
        )

        # load pretrained model
        if args.model_type == "phobert":
            model_path = os.path.join(args.model_name_or_path, "model.bin")
            self.roberta = TFRobertaModel.from_pretrained(
                model_path,
                from_pt=True,
                config=config,
                name="phobert",
            )

        self.transformer_block = TransformerBlock(
            input_dim=args.input_dim,
            max_seq_len=args.max_seq_len,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            embedding_matrix=embedding_matrix,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            name="transformer_block",
        )

        self.aspect_classifiers = []
        for i in range(num_aspects):
            self.aspect_classifiers.append(
                layers.Dense(num_polarities, activation="softmax", name=f"aspect_{i}")
            )

        self.concat = layers.Concatenate(name="concatenate")
        self.dropout = layers.Dropout(args.dropout_rate, name="dropout")

    def build(self, training: bool = True):
        input_ids = self.input_ids
        attention_mask = self.attention_mask
        transformer_mask = self.transformer_mask
        bert_output = self.roberta(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )[2] # [batch_size, seq_len, hidden_dim]
        transformer_output = self.transformer_block(
            input_ids, transformer_mask, training
        )  # [batch_size, seq_len, embed_dim]

        # concat bert_output and transformer_output
        x = self.concat(
            [
                bert_output[-1][:, 0, :],
                bert_output[-2][:, 0, :],
                bert_output[-3][:, 0, :],
                bert_output[-4][:, 0, :],
                transformer_output,
            ]
        )  # [batch_size, 4 * hidden_dim + embed_dim]
        x = self.dropout(x)
        outputs = []
        for aspect_classifier in self.aspect_classifiers:
            aspect_output = aspect_classifier(x)
            outputs.append(aspect_output)

        model = models.Model(
            inputs=[input_ids, attention_mask, transformer_mask],
            outputs=outputs,
            name="aspect_based_model",
        )
        return model
