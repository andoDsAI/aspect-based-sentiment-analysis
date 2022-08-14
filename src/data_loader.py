import copy
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from src.utils import get_aspect_labels, get_polarity_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        aspect_label: (Optional) list. The aspect labels of the example.
        polarity_label: (Optional) list. The polarity labels of the example.
    """

    def __init__(self, guid, words, aspect_label=None, polarity_label=None):
        self.guid = guid
        self.words = words
        self.aspect_label = aspect_label
        self.polarity_label = polarity_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self, input_ids, attention_mask, token_type_ids, aspect_label_ids, polarity_label_ids
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.aspect_label_ids = aspect_label_ids
        self.polarity_label_ids = polarity_label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the dataset"""

    def __init__(self, args):
        self.args = args
        self.aspect_labels = get_aspect_labels(args)
        self.polarity_labels = get_polarity_labels(args)

    def _read_file(self, input_file):
        """Reads a csv file."""
        data = pd.read_csv(input_file)
        reviews = list(data.Review)
        polarities = zip(
            list(data.giai_tri),
            list(data.luu_tru),
            list(data.nha_hang),
            list(data.an_uong),
            list(data.di_chuyen),
            list(data.mua_sam),
        )
        polarity_labels = [list(i) for i in polarities]
        aspect_labels = [[] for _ in range(len(reviews))]
        for i in range(len(reviews)):
            for j in range(len(self.aspect_labels)):
                if polarity_labels[i][j] != 0:
                    aspect_labels[i].append(1)
                else:
                    aspect_labels[i].append(0)

        polarity_labels = np.array(polarity_labels)
        aspect_labels = np.array(aspect_labels)
        return reviews, aspect_labels, polarity_labels

    def _create_examples(self, reviews, aspect_labels, polarity_labels, set_type):
        """Creates examples for the dataset."""
        examples = []
        for i, (review, aspect_label, polarity_label) in enumerate(
            zip(reviews, aspect_labels, polarity_labels)
        ):
            guid = "%s-%s" % (set_type, i)
            # input text
            words = review.split()
            examples.append(
                InputExample(
                    guid=guid,
                    words=words,
                    aspect_label=aspect_label,
                    polarity_label=polarity_label,
                )
            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, mode + ".csv")
        logger.info("LOOKING AT {}".format(data_path))
        reviews, aspect_labels, polarity_labels = self._read_file(data_path)
        return self._create_examples(
            reviews=reviews,
            aspect_labels=aspect_labels,
            polarity_labels=polarity_labels,
            set_type=mode,
        )


def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    pad_token_label_id=-100,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize
        tokens = []
        for word in example.words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        # Input ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(
            len(input_ids), max_seq_len
        )
        assert (
            len(attention_mask) == max_seq_len
        ), "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                aspect_label_ids=example.aspect_label,
                polarity_label_ids=example.polarity_label,
            )
        )
    return features


def load_examples(args, tokenizer, mode):
    processor = JointProcessor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode,
            args.token_level,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len,
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        if mode in [
            "train",
            "train_dev",
            "dev",
            "test",
            "private_test",
            "public_test",
        ]:
            examples = processor.get_examples(mode)
        else:
            raise Exception("For mode {}, Only train, dev, test is available".format(mode))

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(
            examples, args.max_seq_len, tokenizer, pad_token_label_id=pad_token_label_id
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    dataset = TensorDataset(
        torch.tensor([f.input_ids for f in features], dtype=torch.long),
        torch.tensor([f.attention_mask for f in features], dtype=torch.long),
        torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
        torch.tensor([f.aspect_label_ids for f in features], dtype=torch.float32),
        torch.tensor([f.polarity_label_ids for f in features], dtype=torch.long),
    )
    return dataset
