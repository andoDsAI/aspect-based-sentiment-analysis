import logging
import os

import tensorflow_addons as tfa
from tensorflow.keras import callbacks, optimizers

from .callbacks import EarlyStop
from .data_loader import get_embedding_matrix
from .utils import MODEL_CLASSES, get_aspect_labels, get_polarity_labels, scheduler

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.num_aspects = len(get_aspect_labels(self.args))
        self.num_polarities = len(get_polarity_labels(self.args))
        embedding_matrix = get_embedding_matrix(args, corpus=train_dataset["corpus"])

        # Prepare model
        self.config_class, self.model_class = MODEL_CLASSES[args.model_type]
        config_path = os.path.join(args.model_name_or_path, "config.json")
        self.config = self.config_class.from_pretrained(
            config_path, output_hidden_states=True
        )
        self.args.input_dim = embedding_matrix.shape[0]  # vocab size
        self.model = self.model_class(
            config=self.config,
            args=args,
            embedding_matrix=embedding_matrix,
            num_aspects=self.num_aspects,
            num_polarities=self.num_polarities,
        ).build()

        if args.pretrained:
            self.model.load_weights(args.pretrained_path)
        self.model.summary()

    def train(self):
        # train!
        logger.info("***** Training *****")
        logger.info("Num examples = %d", len(self.train_dataset["corpus"]))
        epochs = self.args.epochs
        batch_size = self.args.batch_size
        model_callbacks = [
            callbacks.LearningRateScheduler(scheduler),
            callbacks.ModelCheckpoint(
                os.path.join(self.args.model_dir),
                monitor="val_aspect_f1_score",
                verbose=1,
                mode="max",
                save_weights_only=True,
                save_best_only=True,
            ),
            EarlyStop(
                args=self.args,
                patience=self.args.early_stopping,
                restore_best_weights=True,
                mode="max",
            ),
        ]
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(
                lr=self.args.learning_rate,
                epsilon=self.args.epsilon,
                decay=self.args.weight_decay,
            ),
            metrics=[
                [
                    tfa.metrics.F1Score(
                        num_classes=self.num_aspects,
                        threshold=self.args.threshold,
                        average="micro",
                        name="f1_score",
                    )
                ],
                [
                    tfa.metrics.F1Score(
                        num_classes=self.num_aspects * self.num_polarities,
                        threshold=self.args.threshold,
                        average="micro",
                        name="f1_score",
                    )
                ],
            ],
        )
        
        history = self.model.fit(
            [self.train_dataset["input_ids"], self.train_dataset["attention_mask"], self.train_dataset["transformer_mask"]],
            [self.train_dataset["aspects"], self.train_dataset["polarities"]],
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=(
                [self.dev_dataset["input_ids"], self.dev_dataset["attention_mask"], self.dev_dataset["transformer_mask"]],
                [self.dev_dataset["aspects"], self.dev_dataset["polarities"]],
            ),
            callbacks=model_callbacks,
        )
        self.save_model()
        return history

    def evaluate(self, mode):
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        elif mode == "train":
            dataset = self.train_dataset
        else:
            raise Exception("Only dev and test dataset available")
        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("Num examples = %d", len(dataset["corpus"]))
        self.model.evaluate(
            [dataset["input_ids"], dataset["attention_mask"], dataset["transformer_mask"]], 
            [dataset["aspects"], dataset["polarities"]],
            verbose=1
        )

    def save_model(self):
        # save model checkpoint (overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        logger.info("Saving model checkpoint to %s", self.args.model_dir)
        self.model.save_weights(os.path.join(self.args.model_dir, "weights.h5"))
        logger.info("Saved model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")
        if not os.path.exists(os.path.join(self.args.model_dir, "weights.h5")):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model.load_weights(os.path.join(self.args.model_dir, "weights.h5"))
            logger.info("***** Model Loaded *****")
        except Exception:
            raise Exception("Some model files might be missing...")
