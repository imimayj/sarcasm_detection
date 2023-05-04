import datetime
import json
import math
import os
import pickle
import re
import string
import subprocess

import contractions
import evaluate
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn
import tensorboard
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import transformers
from electra_classifier import ElectraClassifier, SarcasmDataModule, SarcasmDataset
from finetuning_scheduler import FinetuningScheduler
from nltk.corpus import stopwords
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LambdaLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchmetrics import Accuracy, Precision, Recall
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from transformers import (
    AdamW,
    AutoTokenizer,
    DataCollatorWithPadding,
    ElectraConfig,
    ElectraForSequenceClassification,
    ElectraModel,
    ElectraTokenizer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

data_path = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/Sarcasm_Headlines_Dataset_v2.json"
sub_data_path_train = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/train.csv"
sub_data_path_test = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/test.csv"
version_number = 2
sub_version_number = 2
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = f"/workspaces/sarcasm_detection/sarcasm_detection/checkpoints/sarcasm_detection_finetune_ckpt_v{version_number}_{current_time}.ckpt"
sub_checkpoint_path = f"/workspaces/sarcasm_detection/sarcasm_detection/checkpoints/subcat_finetune_ckpt_v{sub_version_number}_{current_time}.ckpt"
checkpoint_directory = os.path.dirname(checkpoint_path)
logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"


# data module
class SubcategoryDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["tweet"]
        labels = self.data[idx]["sarcastic"]
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return encodings["input_ids"].flatten(), encodings["attention_mask"].flatten(), torch.tensor(labels)


class SarcasmSubDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, tokenizer="google/electra-small-discriminator"):
        super().__init__()
        self.train_data_path = data_path[0]
        self.test_data_path = data_path[1]
        self.tokenizer = ElectraTokenizer.from_pretrained(tokenizer)
        self.batch_size = batch_size
        self.collate_fn = self.default_collate_fn

    def remove_twitter_handles(self, text):
        return re.sub(r"@[\w]+", "account_name", text)

    def prepare_data(self):
        col_types = {"tweet": "str", "sarcastic": "int32"}

        df = (
            pd.read_csv(self.train_data_path)
            .drop(
                columns=[
                    "rephrase",
                    "sarcasm",
                    "irony",
                    "satire",
                    "understatement",
                    "overstatement",
                    "rhetorical_question",
                ]
            )
            .astype(col_types)
        )

        df["tweet"] = df["tweet"].apply(self.remove_twitter_handles)

        train_df, val_df = self.split_datasets(df)
        print(train_df.head(10))

        col_types = {"text": "str", "sarcasm": "int32"}

        test_df = (
            pd.read_csv(self.test_data_path)
            .drop(columns=["irony", "satire", "understatement", "overstatement", "rhetorical_question"])
            .astype(col_types)
        )

        test_df["text"] = test_df["text"].apply(self.remove_twitter_handles)

        # print(f"training df length: {len(train_df)}")
        # print(f"Validation DataFrame length: {len(val_df)}")
        # print(f"Test DataFrame length: {len(test_df)}")
        # print(f"total df len: {len(train_df+val_df+test_df)}")
        # print(len(df))

        self.data_train = train_df.to_dict("records")
        self.data_val = val_df.to_dict("records")
        self.data_test = test_df.to_dict("records")

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_dataset = SubcategoryDataset(self.data_train, self.tokenizer)
            self.val_dataset = SubcategoryDataset(self.data_val, self.tokenizer)

        if stage == "test":
            self.test_dataset = SubcategoryDataset(self.data_test, self.tokenizer)

        if stage == "predict":
            self.predict_dataset = SubcategoryDataset(self.data_test, self.tokenizer)

    def steps_per_epoch(self):
        return len(self.train_dataset)

    def split_datasets(self, df):
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        return train_df, val_df

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=6, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=6, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=6, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=6, shuffle=False)

    def default_collate_fn(self, batch):
        input_ids, attention_mask, labels = zip(*batch)
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)
        return input_ids, attention_mask, labels


class CustomElectraClassifier(ElectraClassifier):
    def __init__(self, data_module=None, batch_size=None, num_labels=None, electra_classifier=None, learning_rate=None):
        super().__init__(data_module=data_module, batch_size=batch_size, num_labels=num_labels)

        if electra_classifier is not None:
            self.model = electra_classifier.model
        else:
            raise ValueError("An ElectraClassifier must be instantiated")

        serialiazable_hparams = {"num_labels": num_labels, "batch_size": batch_size, "learning_rate": learning_rate}

        self.save_hyperparameters(serialiazable_hparams)

        # self.model = model_name.from_pretrained(model_name, num_labels=num_labels)
        self.data_module = data_module
        self.num_layers = len(list(self.parameters()))
        self.electra = electra_classifier.model
        self.warmup_steps = None
        self.learning_rate = learning_rate

        # self.additional_layer_1 = electra_classifier.additional_layer_1
        # self.activation = electra_classifier.activation
        # self.classifier = nn.Linear(64, num_labels)
        # self.dropout = nn.Dropout(0)
        # self.predictions = []
        # self.targets = []

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # metrics
        self.train_accuracy = Accuracy(task="binary", num_classes=num_labels)
        self.val_accuracy = Accuracy(task="binary", num_classes=num_labels)
        self.test_accuracy = Accuracy(task="binary", num_classes=num_labels)
        self.train_precision = Precision(task="binary", num_classes=num_labels, average="weighted")
        self.val_precision = Precision(task="binary", num_classes=num_labels, average="weighted")
        self.train_recall = Recall(task="binary", num_classes=num_labels, average="weighted")
        self.val_recall = Recall(task="binary", num_classes=num_labels, average="weighted")
        self.val_f1_score = BinaryF1Score(task="binary", num_classes=num_labels)
        self.f1 = BinaryF1Score(task="binary", num_classes=num_labels, average="macro")

        # for adding smaller networks on top
        # self.dropout = nn.Dropout(0.1)
        # self.fc1 = nn.Linear(self.electra.config.hidden_size, 128)
        # self.fc2 = nn.Linear(128, num_labels)

    @property
    def electra(self):
        return self.model.electra

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloder()

    def predict_dataloader(self):
        return self.data_module.predict_dataloader()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # for adding smaller networks on top
        # outputs = self.electra(input_ids, attention_mask)
        # x = self.activation(self.additional_layer_1(outputs.logits))
        # logits = self.classifier(x)
        # return x

    # def update_classification_head(self, num_labels):
    #     self.classifier = nn.Linear(64, num_labels)

    def on_train_start(self):
        self.train_dataset_len = len(self.train_dataloader().dataset)

    def on_train_batch_start(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)

        # logging
        acc = self.train_accuracy(preds, labels)
        prec = self.train_precision(preds, labels)
        rec = self.train_recall(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_precision", prec, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_recall", rec, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)

        # logging
        acc = self.val_accuracy(preds, labels)
        prec = self.val_precision(preds, labels)
        rec = self.val_recall(preds, labels)
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)
        self.log("val_precision", prec)
        self.log("val_recall", rec)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        acc = self.test_accuracy(preds, labels)
        # prec = self.val_precision(preds, labels)
        # rec = self.val_recall(preds, labels)
        self.predictions.append(preds.detach().cpu())
        self.targets.append(labels.detach().cpu())
        f1_score = self.f1(preds, labels)
        print("logits test_step:", outputs.logits)
        print("predictions test_step:", preds)
        self.log("test_f1", f1_score, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_start(self):
        self.predictions = []
        self.targets = []

    def on_test_epoch_end(self):
        print("predictions on_test_epoch_end:", self.predictions)
        print("targets on_test_epoch_end", self.targets)
        all_preds = torch.cat(self.predictions)
        all_labels = torch.cat(self.targets)
        return {"preds": all_preds, "labels": all_labels}

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = outputs.logits.argmax(dim=-1)
        self.predictions.append(preds.detach().cpu())
        self.targets.append(labels.detach().cpu())
        f1_score = self.f1(preds, labels)
        print("logits test_step:", outputs.logits)
        print("predictions test_step:", preds)
        return preds, f1_score

    def on_predict_start(self):
        self.predictions = []
        self.targets = []

    def collate_fn(self, batch):
        inputs, labels = zip(*batch)
        return torch.stack(inputs), torch.stack(labels)

    @property
    def current_lr(self):
        return self.optimizers().param_groups[0]["lr"]

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.warmup_steps,
        #     num_training_steps=self.trainer.estimated_stepping_batches
        # )

        # scheduler = {
        #     'scheduler': scheduler,
        #     'interval': 'step',
        #     'frequency': 1
        # }
        return optimizer

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_metrics = []
        self.val_metrics = []
        self.batch_train_metrics = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_train_metrics.append(trainer.callback_metrics)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_train_metrics = {}
        for key in self.batch_train_metrics[0].keys():
            epoch_train_metrics[key] = torch.stack([x[key] for x in self.batch_train_metrics]).mean()
        self.train_metrics.append(epoch_train_metrics)
        self.batch_train_metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_metrics.append(trainer.callback_metrics)


def find_latest_checkpoint(version_prefix="sarcasm_detection_finetune_ckpt_v"):
    checkpoints = [
        file for file in os.listdir(checkpoint_directory) if file.startswith(version_prefix) and file.endswith(".ckpt")
    ]
    return (
        max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_directory, x))) if checkpoints else None
    )


sub_data_module = SarcasmSubDataModule(data_path=[sub_data_path_train, sub_data_path_test], batch_size=32)


def load_model(saved_data_module, transfer_data_module):
    latest_checkpoint = find_latest_checkpoint()

    if latest_checkpoint:
        full_checkpoint_path = os.path.join(checkpoint_directory, latest_checkpoint)

        # load model
        loaded_model = ElectraClassifier.load_from_checkpoint(full_checkpoint_path, data_module=saved_data_module)

        # new model with different classification heads
        transfer_model = CustomElectraClassifier(
            electra_classifier=loaded_model,
            data_module=transfer_data_module,
            num_labels=2,
            batch_size=transfer_data_module.batch_size,
            learning_rate=5e-6,
        )

        transfer_model.model.electra.load_state_dict(loaded_model.model.electra.state_dict())

        return transfer_model

    else:
        print("No model checkpoint found")


def fit(model, data_module):
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=logdir, name=f"esubcat_model_v{sub_version_number}")
    metrics_callback = MetricsCallback()
    early_stopping = EarlyStopping("val_loss", patience=5, verbose=True)

    trainer = Trainer(
        max_epochs=100,
        callbacks=[
            lr_monitor,
            metrics_callback,
            early_stopping,
        ],
        logger=logger,
    )

    trainer.fit(model, data_module)

    trainer.save_checkpoint(sub_checkpoint_path)
    return model, data_module


def test(model, data_module):
    trainer = Trainer()
    # testing

    test_result = trainer.test(model, data_module)
    print(test_result)


def get_f1_scores(predict_result):
    f1_scores = [f1_score for (_batch_preds, f1_score) in predict_result]
    f1_scores = torch.mean(torch.stack(f1_scores))
    return f1_scores


# predicting


def predict(model, data_module):
    trainer = Trainer()

    predict_result = trainer.predict(model, data_module)

    print(predict_result)

    f1_scores = get_f1_scores(predict_result)
    print(f"f1 score for predictions: {torch.IntTensor.item(f1_scores)}")


def launch_tensorboard(logdir):
    command = f"tensorboard --logdir={logdir}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return process


def main():
    saved_data_module = SarcasmDataModule(data_path=data_path, batch_size=16)
    sub_data_module = SarcasmSubDataModule(data_path=[sub_data_path_train, sub_data_path_test], batch_size=32)
    transfer_model = load_model(saved_data_module=saved_data_module, transfer_data_module=sub_data_module)
    if transfer_model is not None:
        logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"
        tensorboard_process = launch_tensorboard(logdir)
        transfer_model = fit(transfer_model, sub_data_module)
        tensorboard_process.terminate()
    else:
        print("failed to load the transfer model")


if __name__ == "__main__":
    main()
