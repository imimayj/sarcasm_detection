import datetime
import os
import subprocess

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import tensorboard
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateFinder, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.nn.functional import cross_entropy
from torch.optim import AdamW, RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchmetrics import Accuracy, Precision, Recall
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from transformers import (
    AdamW,
    ElectraForSequenceClassification,
    ElectraTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from transformers.models.electra.modeling_electra import ElectraClassificationHead

data_path = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/Sarcasm_Headlines_Dataset_v2.json"
sub_data_path_train = "/workspaces/sarcasm_detection/notebooks/project_data/train.csv"
sub_data_path_test = "/workspaces/sarcasm_detection/notebooks/project_data/test.csv"
version_number = 6
sub_version_number = 0
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = f"/workspaces/sarcasm_detection/sarcasm_detection/checkpoints/sarcasm_detection_finetune_ckpt_v{version_number}_{current_time}.ckpt"
sub_checkpoint_path = f"/workspaces/sarcasm_detection/notebooks/checkpoints/subcat_finetune_ckpt_v{sub_version_number}_{current_time}.ckpt"
checkpoint_directory = os.path.dirname(checkpoint_path)
logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"
save_directory = (
    f"/workspaces/sarcasm_detection/sarcasm_detection/saved_models/sarcasm_model_v{version_number}_{current_time}"
)


class SarcasmDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["headline"]
        labels = self.data[idx]["is_sarcastic"]
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return (
            encodings["input_ids"].flatten(),
            encodings["attention_mask"].flatten(),
            torch.tensor(labels),
        )


class SarcasmDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, tokenizer="google/electra-small-discriminator"):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = ElectraTokenizer.from_pretrained(tokenizer)
        self.batch_size = batch_size
        self.collate_fn = self.default_collate_fn

    def prepare_data(self):
        col_types = {"headline": "str", "is_sarcastic": "int32"}

        df = pd.read_json(self.data_path, lines=True).drop(columns=["article_link"]).astype(col_types)

        train_df, val_df, test_df = self.split_datasets(df)

        print("training dataset class distribution:")
        for label in ["is_sarcastic"]:
            train_class_counts = train_df[label].value_counts()
            print(f"{label}:")
            print(train_class_counts)

        print("validation dataset class distribution:")
        for label in ["is_sarcastic"]:
            val_class_counts = val_df[label].value_counts()
            print(f"{label}:")
            print(val_class_counts)

        print("test dataset class distribution:")
        for label in ["is_sarcastic"]:
            test_class_counts = test_df[label].value_counts()
            print(f"{label}:")
            print(test_class_counts)

        self.data_train = train_df.to_dict("records")
        self.data_val = val_df.to_dict("records")
        self.data_test = test_df.to_dict("records")

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_dataset = SarcasmDataset(self.data_train, self.tokenizer)
            self.val_dataset = SarcasmDataset(self.data_val, self.tokenizer)

        if stage == "test":
            self.test_dataset = SarcasmDataset(self.data_test, self.tokenizer)

        if stage == "predict":
            self.predict_dataset = SarcasmDataset(self.data_test, self.tokenizer)

    def steps_per_epoch(self):
        return len(self.train_dataset)

    def split_datasets(self, df):
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)
        return train_df, val_df, test_df

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=6, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=6, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=6, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            shuffle=False,
        )

    def default_collate_fn(self, batch):
        input_ids, attention_mask, labels = zip(*batch)
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)
        return input_ids, attention_mask, labels


class ElectraClassifier(pl.LightningModule):
    def __init__(
        self,
        data_module,
        batch_size,
        model_name="google/electra-small-discriminator",
        num_labels=None,
        learning_rate=2e-5,
    ):
        super().__init__()

        serialiazable_hparams = {"num_labels": num_labels, "batch_size": batch_size, "learning_rate": learning_rate}
        self.save_hyperparameters(serialiazable_hparams)
        self.model_name_or_path = model_name
        self.model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.data_module = data_module
        self.num_layers = self.model.config.num_hidden_layers
        self.warmup_steps = 10000
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_labels = num_labels or data_module.num_labels
        self.best_val_performance = -np.inf
        self.epochs_since_best_performance = 0
        self.patience = 2
        self.current_unfreeze_idx = len(list(self.model.electra.parameters())) - 1
        self.unfreeze_step = 1
        # self.epoch_train_losses = []

        for param in self.model.electra.parameters():
            param.requires_grad = False

        # metrics
        self.val_loss = None
        self.train_accuracy = Accuracy(task="binary", num_classes=num_labels)
        self.val_accuracy = Accuracy(task="binary", num_classes=num_labels)
        self.test_accuracy = Accuracy(task="binary", num_classes=num_labels)
        self.train_precision = Precision(task="binary", num_classes=num_labels, average="weighted")
        self.val_precision = Precision(task="binary", num_classes=num_labels, average="weighted")
        self.train_recall = Recall(task="binary", num_classes=num_labels, average="weighted")
        self.val_recall = Recall(task="binary", num_classes=num_labels, average="weighted")
        self.val_f1_score = BinaryF1Score(task="binary", num_classes=num_labels)
        self.f1 = BinaryF1Score(task="binary", num_classes=num_labels, average="macro")

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

    def on_train_start(self):
        self.train_dataset_len = len(self.train_dataloader().dataset)

    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step == self.warmup_steps:
            optimizer = self.optimizers()
            freeze_idx = int(len(list(self.model.electra.parameters())) * 0.98)

            for idx, param_group in enumerate(optimizer.param_groups):
                if self.global_step < self.warmup_steps:
                    if idx < freeze_idx:
                        param_group["lr"] = 0

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
        self.log("learning_rate", self.learning_rate, on_step=True, on_epoch=True, prog_bar=True)

        # self.epoch_train_losses.append(loss)
        return loss

    # def on_train_epoch_end(self):
    #     avg_train_loss = torch.stack(self.epoch_train_losses).mean()
    #     self.log("train_epoch_loss", avg_train_loss)
    #     self.epoch_train_losses = []

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        # logging
        acc = self.val_accuracy(preds, labels)
        prec = self.val_precision(preds, labels)
        rec = self.val_recall(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recall", rec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("learning_rate", self.learning_rate, on_step=True, on_epoch=True, prog_bar=True)

        self.val_loss = loss.item()

    def on_validation_epoch_end(self):
        print("on_validation_epoch_end called")
        current_val_performance = self.val_loss
        improvement_threshold = 0.01
        improvement = self.best_val_performance - current_val_performance
        if current_val_performance > self.best_val_performance:
            self.best_val_performance = current_val_performance
            self.epochs_since_best_performance = 0
        else:
            self.epochs_since_best_performance += 1

        if improvement >= improvement_threshold and self.global_step >= self.warmup_steps:
            self.unfreeze_next_layer()

    def unfreeze_next_layer(self):
        num_params = len(list(self.model.electra.parameters()))
        freeze_idx = int(num_params * 0.98)
        unfreeze_idx = self.current_unfreeze_idx

        print(f"unfreeze_idx: {unfreeze_idx}, freeze_idx: {freeze_idx}")

        if self.current_unfreeze_idx >= 0 and self.current_unfreeze_idx >= freeze_idx:
            param = list(self.model.electra.parameters())[unfreeze_idx]
            param.requires_grad = True
            print(f"unfroze layer idx {self.current_unfreeze_idx - 1}")
            self.current_unfreeze_idx -= self.unfreeze_step

        print("layer-wise requires_grad status:")
        for idx, param in enumerate(self.model.electra.parameters()):
            print(f"layer{idx}: {param.requires_grad}")

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        acc = self.test_accuracy(preds, labels)
        self.predictions.append(preds.detach().cpu())
        self.targets.append(labels.detach().cpu())
        f1_score = self.f1(preds, labels)
        # print("logits test_step:", outputs.logits)
        # print("predictions test_step:", preds)
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
    def config(self):
        return self.model.config

    @property
    def current_lr(self):
        return self.optimizers().param_groups[0]["lr"]

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(), lr=self.hparams.learning_rate)

        num_training_steps = len(self.train_dataloader()) * self.trainer.max_epochs
        print(f"num_warmup_steps: {self.warmup_steps}, num_training_steps: {num_training_steps}")
        # replacing num_training_steps = self.trainer.estimated_stepping_batches

        T_0 = num_training_steps - self.warmup_steps
        T_mult = 1
        eta_min = 0

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

        scheduler_with_warmup = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "warmup_steps": self.warmup_steps,
            "warmup_start_lr": 0,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_with_warmup,
            "monitor": "train_loss",
        }

    def update_classification_head(self, num_labels):
        self.model.classifier = ElectraClassificationHead(self.model.config, num_labels)
        self.num_labels = num_labels

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# metrics plotting


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


def fit(model, data_module):
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=logdir, name=f"electra_model_v{version_number}")
    metrics_callback = MetricsCallback()
    early_stopping = EarlyStopping("val_loss", patience=5, verbose=True)

    trainer = Trainer(
        max_epochs=10000,
        callbacks=[
            lr_monitor,
            metrics_callback,
            early_stopping,
        ],
        logger=logger,
    )

    trainer.fit(model, data_module)

    trainer.save_checkpoint(checkpoint_path)
    base_model = model.model.base_model
    base_model.save_pretrained(save_directory)
    return model


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
    logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"
    tensorboard_process = launch_tensorboard(logdir)
    data_module = SarcasmDataModule(data_path=data_path, batch_size=16)
    model = ElectraClassifier(data_module=data_module, batch_size=data_module.batch_size, num_labels=2)
    model = fit(model, data_module)

    tensorboard_process.terminate()

    model = test(model, data_module)


if __name__ == "__main__":
    main()

# %tensorboard --logdir=workspaces/sarcasm_detection/sarcasm_detection/tb_logs/
