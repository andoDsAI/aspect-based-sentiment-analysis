import argparse
import os
import warnings

from src.data_loader import load_samples
from src.trainer import Trainer
from src.utils import MODEL_CLASSES, MODEL_PATH_MAP, draw_history, init_logger, seed_everything

warnings.filterwarnings("ignore")


def main(args):
    init_logger()
    seed_everything(args)

    train_dataset = load_samples(args, data_type="train")
    dev_dataset = load_samples(args, data_type="dev")
    test_dataset = load_samples(args, data_type="test")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.do_train:
        history = trainer.train()
        if args.plot_result:
            draw_history(history)

    if args.eval_train:
        trainer.load_model()
        trainer.evaluate("train")

    if args.eval_dev:
        trainer.load_model()
        trainer.evaluate("dev")

    if args.eval_test:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir", default="./trained_models", type=str, help="Path to save, load model"
    )
    parser.add_argument("--log_dir", default="./logs", type=str, help="Path to log directory")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument(
        "--aspect_label_file", default="aspect_label.json", type=str, help="Aspect Label file"
    )
    parser.add_argument(
        "--polarity_label_file",
        default="polarity_label.json",
        type=str,
        help="Polarity Label file",
    )
    parser.add_argument(
        "--model_type",
        default="phobert",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--plot_result",
        action="store_true",
        help="Whether to plot the training result",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--eval_train", action="store_true", help="Whether to run eval on the train set."
    )
    parser.add_argument(
        "--eval_dev", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--eval_test", action="store_true", help="Whether to run eval on the test set."
    )

    # training parameters
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training.")
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument("--epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument(
        "--aspect_coef", type=float, default=0.5, help="Coefficient for the aspect loss."
    )
    parser.add_argument(
        "--max_seq_len",
        default=96,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=400, help="Embedding size for bert output"
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden size for bert output")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of attention layers")
    parser.add_argument(
        "--dropout_rate", default=0.4, type=float, help="Dropout for fully-connected layers"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold of sigmoid function"
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=15,
        help="Number of un-increased validation step to wait for early stopping",
    )

    # Init pretrained
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Whether to init model from pretrained base model",
    )
    parser.add_argument(
        "--pretrained_path", default="./trained_models", type=str, help="The pretrained model path"
    )

    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
