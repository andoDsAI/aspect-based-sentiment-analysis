import logging
import os

import numpy as np
import pandas as pd
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE
from gensim.models import Word2Vec
from tqdm import tqdm

from src.utils import get_aspect_labels, get_polarity_labels
from vncorenlp import VnCoreNLP

logger = logging.getLogger(__name__)


class BPE:
    bpe_codes = "PhoBERT_base_transformers/bpe.codes"


args = BPE()
bpe = fastBPE(args)
vocab = Dictionary()
vocab.add_from_file("PhoBERT_base_transformers/dict.txt")
rdrsegmenter_path = "vncorenlp/VnCoreNLP-1.1.1.jar"


def load_data(args, data_type: str = "train"):
    aspects_label = get_aspect_labels(args)
    polarities_label = get_polarity_labels(args)
    num_aspect = len(aspects_label)
    num_sentiment = len(polarities_label)
    df = pd.read_csv(os.path.join(args.data_dir, data_type + ".csv"))

    texts = []
    aspects = []
    polarities = []
    for _, row in tqdm(df.iterrows(), desc=f"Loading {data_type} data...", total=len(df)):
        texts.append(row["text"])
        aspects.append(np.zeros(num_aspect))
        polarities.append(np.zeros(num_aspect * num_sentiment))
        for key in aspects_label.keys():
            if row[key] != 0:
                aspects[-1][aspects_label[key]] = 1
                polarities[-1][aspects_label[key] * num_sentiment + row[key] - 1] = 1

    aspects = np.array(aspects)
    polarities = np.array(polarities)
    return texts, aspects, polarities


def convert_lines_to_features(args, lines):
    rdrsegmenter = VnCoreNLP(rdrsegmenter_path, annotators="wseg", max_heap_size="-Xmx500m")
    max_seq_len = args.max_seq_len
    # initialize the numpy arrays
    outputs = np.zeros((len(lines), max_seq_len), dtype=np.int32)
    attention_mask = np.ones((len(lines), max_seq_len), dtype=np.int32)
    transformer_mask = np.ones((len(lines), max_seq_len, max_seq_len), dtype=np.int32)

    pad_id = 1
    eos_id = 2
    corpus = []
    for idx, row in tqdm(
        enumerate(lines), total=len(lines), desc="Converting lines to features..."
    ):
        row = " ".join([" ".join(sent) for sent in rdrsegmenter.tokenize(row)])
        # byte pair encoding(bpe)
        sub_words = "<s> " + bpe.encode(row) + " </s>"
        corpus.append(sub_words.split())
        input_ids = (
            vocab.encode_line(sub_words, append_eos=False, add_if_not_exist=False).long().tolist()
        )
        # padding
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            input_ids[-1] = eos_id
        else:
            length_ = len(input_ids)
            input_ids = input_ids + [pad_id] * (max_seq_len - len(input_ids))
            mask = [0 if i == 1 else 1 for i in input_ids]
            attention_mask[idx, :] = np.array(mask)
            transformer_mask[idx, :, length_:] = 0
            transformer_mask[idx, length_:, :] = 0
        outputs[idx, :] = np.array(input_ids)
    rdrsegmenter.close()
    return outputs, attention_mask, transformer_mask, corpus


def load_samples(args, data_type):
    # load data features from dataset file
    if data_type in [
        "train",
        "dev",
        "test",
    ]:
        lines, aspects, polarities = load_data(args, data_type)
    else:
        raise Exception("For mode {}, Only train, dev, test is available".format(data_type))
    input_ids, attention_mask, transformer_mask, corpus = convert_lines_to_features(args, lines)
    del lines
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "transformer_mask": transformer_mask,
        "aspects": aspects,
        "polarities": polarities,
        "corpus": corpus,
    }


def get_embedding_matrix(args, corpus):
    # word2vec model
    word2vec = Word2Vec(
        sentences=corpus, size=args.embed_dim, window=10, min_count=1, workers=5, sg=1
    )
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, args.embed_dim))
    for word in word2vec.wv.vocab:
        try:
            i = vocab.indices[word]
        except Exception:
            i = vocab.indices["<unk>"]
        embedding_matrix[i] = word2vec.wv[word]
    return embedding_matrix
