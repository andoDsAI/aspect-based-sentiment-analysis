import logging
import os
import random

import numpy as np
import torch
from transformers import (
    AdamW,
    AutoTokenizer,
    RobertaConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)

from network import AspectModel

MODEL_CLASSES = {
    "xlmr": (XLMRobertaConfig, AspectModel, XLMRobertaTokenizer),
    "xlmr-fast": (XLMRobertaConfig, AspectModel, XLMRobertaTokenizerFast),
    "phobert": (RobertaConfig, AspectModel, AutoTokenizer),
}

MODEL_PATH_MAP = {
    "xlmr": "xlm-roberta-base",
    "phobert": "vinai/phobert-base",
}


def seed_everything(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def get_optimizer(model, args, t_total):
    # Prepare optimizer and schedule (linear warmup and decay)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                param for name, param in param_optimizer if not any(nd in name for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                param for name, param in param_optimizer if any(nd in name for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_training_steps=t_total, num_warmup_steps=args.num_warmup_steps
    )

    return optimizer, scheduler


def get_aspect_labels(args):
    aspect_path = os.path.join(args.data_dir, args.aspect_label_file)
    with open(aspect_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        f.close()

    aspect_list = [row.strip() for row in data]
    return aspect_list


def get_polarity_labels(args):
    polarity_path = os.path.join(args.data_dir, args.polarity_label_file)
    with open(polarity_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        f.close()

    polarity_list = [row.strip() for row in data]
    return polarity_list


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)
