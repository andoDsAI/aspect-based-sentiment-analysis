import logging
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from src.early_stopping import EarlyStopping
from src.eval_helper import compute_metrics
from src.utils import MODEL_CLASSES, get_aspect_labels, get_optimizer, get_polarity_labels

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.aspect_label_list = get_aspect_labels(self.args)
        self.polarity_label_lst = get_polarity_labels(self.args)

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

        if args.pretrained:
            self.model = self.model_class.from_pretrained(
                args.pretrained_path,
                args=args,
                aspect_label_list=self.aspect_label_list,
                polarity_label_lst=self.polarity_label_lst,
            )
        else:
            self.config = self.config_class.from_pretrained(args.model_name_or_path)
            self.model = self.model_class.from_pretrained(
                args.model_name_or_path,
                config=self.config,
                args=args,
                aspect_label_list=self.aspect_label_list,
                polarity_label_lst=self.polarity_label_lst,
            )

        # GPU or CPU
        if torch.cuda.is_available() and not args.no_cuda:
            self.device = "cuda"
            torch.cuda.set_device(self.args.gpu_id)
            print(self.args.gpu_id)
            print(torch.cuda.current_device())
        else:
            self.device = "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size
        )
        writer = SummaryWriter(log_dir=self.args.log_dir)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps
                // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )

        optimizer, scheduler = get_optimizer(self.model, self.args, t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("Num examples = %d", len(self.train_dataset))
        logger.info("Num Epochs = %d", self.args.num_train_epochs)
        logger.info("Total train batch size = %d", self.args.train_batch_size)
        logger.info("Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("Total optimization steps = %d", t_total)
        logger.info("Logging steps = %d", self.args.logging_steps)
        logger.info("Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)

            for step, batch in enumerate(epoch_iterator):
                self.model.train()

                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "aspect_label_ids": batch[3],
                    "polarity_label_ids": batch[4],
                }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]

                outputs = self.model(**inputs)
                loss = outputs[0]

                # clip gradient
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        print("\nTuning metrics:", self.args.tuning_metric)
                        results = self.evaluate("train")
                        writer.add_scalar("Loss/train", results["loss"], _)
                        writer.add_scalar("Aspect F1/train", results["aspect_f1"], _)
                        writer.add_scalar("Polarity F1/train", results["polarity_f1"], _)
                        writer.add_scalar(
                            "Mean Aspect Polarity/train", results["mean_aspect_polarity"], _
                        )
                        writer.add_scalar(
                            "Competition Score/train", results["competition_score"], _
                        )

                        results = self.evaluate("dev")
                        writer.add_scalar("Loss/dev", results["loss"], _)
                        writer.add_scalar("Aspect F1/dev", results["aspect_f1"], _)
                        writer.add_scalar("Polarity F1/dev", results["polarity_f1"], _)
                        writer.add_scalar(
                            "Mean Aspect Polarity/dev", results["mean_aspect_polarity"], _
                        )
                        writer.add_scalar("Competition Score/dev", results["competition_score"], _)

                        early_stopping(results[self.args.tuning_metric], self.model, self.args)
                        if early_stopping.early_stop:
                            print("Early stopping")
                            break

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step or early_stopping.early_stop:
                train_iterator.close()
                break

            writer.add_scalar("Loss/train", tr_loss / global_step, _)

        return global_step, tr_loss / global_step

    def write_evaluation_result(self, out_file, results):
        out_file = self.args.model_dir + "/" + out_file
        w = open(out_file, "w", encoding="utf-8")
        w.write("***** Eval results *****\n")
        for key in sorted(results.keys()):
            to_write = " {key} = {value}".format(key=key, value=str(results[key]))
            w.write(to_write)
            w.write("\n")
        w.close()

    def evaluate(self, mode):
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        elif mode == "train":
            dataset = self.train_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size
        )

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("Num examples = %d", len(dataset))
        logger.info("Batch size = %d", self.args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0

        aspect_pred_ids = None
        aspect_label_ids = None
        polarity_pred_ids = None
        polarity_label_ids = None
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "aspect_label_ids": batch[3],
                    "polarity_label_ids": batch[4],
                }
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]

                tmp_eval_loss, aspect_outputs, polarity_outputs = self.model(**inputs)
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Aspect prediction
            if aspect_pred_ids is None:
                aspect_pred_ids = aspect_outputs.detach().cpu().numpy()
                aspect_label_ids = inputs["aspect_label_ids"].detach().cpu().numpy()
            else:
                aspect_pred_ids = np.append(
                    aspect_pred_ids, aspect_outputs.detach().cpu().numpy(), axis=0
                )
                aspect_label_ids = np.append(
                    aspect_label_ids, inputs["aspect_label_ids"].detach().cpu().numpy(), axis=0
                )

            # Polarity prediction
            if polarity_pred_ids is None:
                polarity_pred_ids = polarity_outputs.detach().cpu().numpy()
                polarity_label_ids = inputs["polarity_label_ids"].detach().cpu().numpy()
            else:
                polarity_pred_ids = np.append(
                    polarity_pred_ids, polarity_outputs.detach().cpu().numpy(), axis=0
                )
                polarity_label_ids = np.append(
                    polarity_label_ids, inputs["polarity_label_ids"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        # Aspect result
        aspect_preds = [[] for _ in range(aspect_pred_ids.shape[0])]
        for i in range(aspect_pred_ids.shape[0]):
            for j in range(aspect_pred_ids.shape[1]):
                if aspect_pred_ids[i][j] >= self.args.threshold:
                    aspect_preds[i].append(1)
                else:
                    aspect_preds[i].append(0)

        # Polarity results
        polarity_pred_ids = np.argmax(polarity_pred_ids, axis=2)
        polarity_preds = [[] for _ in range(polarity_pred_ids.shape[0])]
        for i in range(polarity_pred_ids.shape[0]):
            for j in range(polarity_pred_ids.shape[1]):
                if aspect_pred_ids[i][j] == 1:
                    polarity_preds[i][j] = polarity_pred_ids[i][j]
                else:
                    polarity_preds[i][j] = 0
        metrics_result = compute_metrics(
            aspect_preds, aspect_label_ids, polarity_preds, polarity_label_ids
        )
        results.update(metrics_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("%s =\n %s", key, str(results[key]))

        if mode == "test":
            self.write_evaluation_result("eval_test_results.txt", results)
        elif mode == "dev":
            self.write_evaluation_result("eval_dev_results.txt", results)
        elif mode == "train":
            self.write_evaluation_result("eval_train_results.txt", results)
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")
        try:
            self.model = self.model_class.from_pretrained(
                self.args.model_dir,
                args=self.args,
                aspect_label_list=self.aspect_label_list,
                polarity_label_lst=self.polarity_label_lst,
            )
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except Exception:
            raise Exception("Some model files might be missing...")