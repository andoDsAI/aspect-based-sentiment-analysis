import json
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from transformers import RobertaConfig

from network import AspectBasedModel

MODEL_CLASSES = {
    "phobert": (RobertaConfig, AspectBasedModel),
}

MODEL_PATH_MAP = {
    "phobert": "PhoBERT_base_transformers",
}


def seed_everything(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def scheduler(epoch, lr):
    return lr * 0.5 ** (epoch // 9)


def get_aspect_labels(args):
    aspect_path = os.path.join(args.data_dir, args.aspect_label_file)
    with open(aspect_path, "r", encoding="utf-8") as f:
        aspects = json.loads(f.read())
        f.close()

    return aspects


def get_polarity_labels(args):
    polarity_path = os.path.join(args.data_dir, args.polarity_label_file)
    with open(polarity_path, "r", encoding="utf-8") as f:
        polarities = json.loads(f.read())
        f.close()
    return polarities


def draw_history(history):
    # aspect f1-score
    plt.plot(history.history["aspect_f1_score"])
    plt.plot(history.history["val_aspect_f1_score"])
    plt.title("Model aspect f1-score")
    plt.ylabel("f1-score")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    # polarity f1-score
    plt.plot(history.history["polarity_f1_score"])
    plt.plot(history.history["val_polarity_f1_score"])
    plt.title("Model polarity f1-score")
    plt.ylabel("f1-score")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    # loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
