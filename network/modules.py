import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers


def get_angles(pos, angle, dim):
    """Get angles for positional encoding."""
    angle_rates = 1 / np.power(10000, (2 * (angle // 2)) / np.float32(dim))
    return pos * angle_rates


def positional_encoding(position, dim):
    """Positional encoding."""
    pos_encoding = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(dim)[np.newaxis, :], dim
    )
    # apply sin to even indices in the array; 2i
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
    # apply cos to odd indices in the array; 2i + 1
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
    return pos_encoding


class EncoderLayer(layers.Layer):
    """Encoder layer."""

    def __init__(self, input_dim, num_heads, hidden_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.multihead_attn = tfa.layers.MultiHeadAttention(
            head_size=input_dim // num_heads, num_heads=num_heads, dropout=dropout_rate
        )
        self.dropout_1 = layers.Dropout(dropout_rate)
        self.dropout_2 = layers.Dropout(dropout_rate)

        self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)

        self.fc = keras.Sequential(
            [layers.Dense(hidden_dim, activation="relu"), layers.Dense(input_dim)]
        )

    def call(self, x, attention_mask, training: bool):
        attn_output = self.multihead_attn(
            [x, x], mask=attention_mask
        )  # (batch_size, input_seq_len, input_size)
        attn_output = self.dropout_1(
            attn_output, training=training
        )  # (batch_size, input_seq_len, input_size)

        output_1 = self.layer_norm_1(x + attn_output)  # (batch_size, input_seq_len, input_size)
        fc_output = self.fc(output_1)
        fc_output = self.dropout_2(
            fc_output, training=training
        )  # (batch_size, input_seq_len, input_size)
        output = self.layer_norm_2(output_1 + fc_output)  # (batch_size, input_seq_len, input_size)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "num_heads": self.num_heads,
                "hidden_size": self.hidden_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class TokenPositionEmbedding(layers.Layer):
    """Embedding layer."""

    def __init__(self, input_dim, max_seq_len, embed_dim, embedding_matrix, **kwargs):
        super(TokenPositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.embedding_matrix = embedding_matrix

        self.token_embedding = layers.Embedding(
            input_dim=input_dim,
            output_dim=embed_dim,
            weights=[embedding_matrix],
            trainable=True,
            name="token_embedding",
        )
        self.pos_embedding = layers.Embedding(
            input_dim=max_seq_len,
            output_dim=embed_dim,
            weights=[positional_encoding(max_seq_len, embed_dim)],
            trainable=False,
            name="pos_embedding",
        )

    def call(self, x):
        positions = tf.range(start=0, limit=self.max_seq_len, delta=1)
        positions = self.pos_embedding(positions)
        tokens = self.token_embedding(x)
        return tokens + positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "max_seq_len": self.max_seq_len,
                "embed_dim": self.embed_dim,
                "embedding_matrix": self.embedding_matrix,
            }
        )
        return config


class TransformerBlock(layers.Layer):
    """Transformer block."""

    def __init__(
        self,
        input_dim,
        max_seq_len,
        embed_dim,
        hidden_dim,
        num_heads,
        embedding_matrix,
        num_layers,
        dropout_rate=0.1,
        **kwargs
    ):
        super(TransformerBlock, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.embedding_matrix = embedding_matrix
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = TokenPositionEmbedding(
            input_dim, max_seq_len, embed_dim, embedding_matrix
        )
        self.enc_layers = [
            EncoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ]
        self.global_average_pooling = layers.GlobalAveragePooling1D()

    def call(self, input_ids, mask, training):
        x = self.embedding(input_ids)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training)
        x = self.global_average_pooling(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "max_seq_len": self.max_seq_len,
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "embedding_matrix": self.embedding_matrix,
                "num_layers": self.num_layers,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
