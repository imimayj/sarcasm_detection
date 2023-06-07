import datetime
import json
import math
import os
import pickle
import random
import re
import string
import subprocess
from pathlib import Path

import contractions
import evaluate
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn
import tensorboard
import textattack
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import transformers
from electra_classifier import (ElectraClassifier, SarcasmDataModule,
                                SarcasmDataset)
from finetuning_scheduler import FinetuningScheduler
from nltk.corpus import stopwords, wordnet
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelSummary)
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from textattack.augmentation import (Augmenter, CharSwapAugmenter,
                                     DeletionAugmenter, EasyDataAugmenter,
                                     WordNetAugmenter)
from textattack.constraints.pre_transformation import (RepeatModification,
                                                       StopwordModification)
from textattack.transformations import (BackTranslation,
                                        CompositeTransformation,
                                        WordSwapEmbedding, WordSwapExtend,
                                        WordSwapQWERTY,
                                        WordSwapRandomCharacterDeletion,
                                        WordSwapRandomCharacterInsertion)
# from textattack.transformations.sentence_transformations import BackTranslation
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.optim import AdamW, RAdam
from torch.optim.lr_scheduler import (CosineAnnealingWarmRestarts,
                                      ReduceLROnPlateau, StepLR)
from torch.utils.data import (DataLoader, Dataset, TensorDataset,
                              WeightedRandomSampler, random_split)
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from transformers import (  # cosine_schedule_with_warmup,; get_linear_schedule_with_warmup,
    AdamW, AutoTokenizer, DataCollatorWithPadding, ElectraConfig,
    ElectraForSequenceClassification, ElectraModel, ElectraTokenizer,
    TrainingArguments)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.electra.modeling_electra import \
    ElectraClassificationHead

data_path = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/Sarcasm_Headlines_Dataset_v2.json"
sub_data_path_train = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/train.csv"
sub_data_path_test = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/test.csv"
version_number = 8
sub_version_number = 10
non_sarc_vnum = 2
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = f"/workspaces/sarcasm_detection/sarcasm_detection/checkpoints/fine_tuned/sarcasm_detection_finetune_ckpt_v{version_number}_{current_time}.ckpt"
new_ckpt_pth = f"/workspaces/sarcasm_detection/sarcasm_detection/checkpoints/base/subcat_finetune_ckpt_v{non_sarc_vnum}_{current_time}.ckpt"
checkpoint_directory = os.path.dirname(checkpoint_path)
logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"
save_directory = (
    f"/workspaces/sarcasm_detection/sarcasm_detection/saved_models/sarcasm_model_v{version_number}_{current_time}"
)


class SubcategoryDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # text = self.data[idx]["tweet"]
        # keys = ["sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]
        # labels = [self.data[idx][key] for key in keys]
        # encodings = self.tokenizer(
        #     text,
        #     max_length=self.max_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_attention_mask=True,
        #     return_token_type_ids=False,
        #     return_tensors="pt",
        # )
        # return encodings["input_ids"].squeeze(), encodings["attention_mask"].squeeze(), torch.tensor(labels)

        text_key = "tweet" if "tweet" in self.data[idx] else "text"
        text = self.data[idx][text_key]

        # text = self.data[idx]["tweet"]
        keys = ["sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]
        labels = [self.data[idx][key] for key in keys]
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
            encodings["input_ids"].squeeze(),
            encodings["attention_mask"].squeeze(),
            torch.tensor(labels).unsqueeze(0),
        )


class SarcasmSubDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, tokenizer="google/electra-small-discriminator", stage=None):
        super().__init__()
        self.train_data_path = data_path[0]
        self.test_data_path = data_path[1]
        self.tokenizer = ElectraTokenizer.from_pretrained(tokenizer)
        self.batch_size = batch_size
        self.collate_fn = self.default_collate_fn
        self.stage = stage
        transformations = CompositeTransformation(
            [WordSwapExtend(), WordSwapRandomCharacterInsertion(), WordSwapRandomCharacterDeletion()]
        )
        constraints = [RepeatModification(), StopwordModification()]
        self.sarcasm_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=5,
            pct_words_to_swap=0.02,
        )
        self.satire_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=50,
            pct_words_to_swap=0.025,
        )

        self.irony_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=25,
            pct_words_to_swap=0.02,
        )

        self.under_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=50,
            pct_words_to_swap=0.02,
        )

        self.over_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=50,
            pct_words_to_swap=0.02,
        )

        self.rhetq_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=50,
            pct_words_to_swap=0.02,
        )

    def remove_twitter_handles(self, text):
        return re.sub(r"@[\w]+", "account_name", text)

    def remove_urls(self, text):
        return re.sub(r"http\S+", " ", text)

    def remove_contractions(self, text):
        return contractions.fix(text)

    def augment_sarcasm_text(self, text):
        augmented_text = self.sarcasm_augmenter.augment(text)
        return augmented_text

    def augment_irony_text(self, text):
        augmented_text = self.irony_augmenter.augment(text)
        return augmented_text

    def augment_satire_text(self, text):
        augmented_text = self.satire_augmenter.augment(text)
        return augmented_text

    def augment_under_text(self, text):
        augmented_text = self.under_augmenter.augment(text)
        return augmented_text

    def augment_over_text(self, text):
        augmented_text = self.over_augmenter.augment(text)
        return augmented_text

    def augment_rhetq_text(self, text):
        augmented_text = self.rhetq_augmenter.augment(text)
        return augmented_text

    def augment_sarcasm_data(self, df):
        df["tweet"] = df["tweet"].apply(self.augment_sarcasm_text)
        df_0_augmented = df.explode("tweet").reset_index(drop=True)
        return df_0_augmented

    def augment_irony_data(self, df):
        df["tweet"] = df["tweet"].apply(self.augment_irony_text)
        df_0_augmented = df.explode("tweet").reset_index(drop=True)
        return df_0_augmented

    def augment_satire_data(self, df):
        df["tweet"] = df["tweet"].apply(self.augment_satire_text)
        df_0_augmented = df.explode("tweet").reset_index(drop=True)
        return df_0_augmented

    def augment_under_data(self, df):
        df["tweet"] = df["tweet"].apply(self.augment_under_text)
        df_0_augmented = df.explode("tweet").reset_index(drop=True)
        return df_0_augmented

    def augment_over_data(self, df):
        df["tweet"] = df["tweet"].apply(self.augment_over_text)
        df_0_augmented = df.explode("tweet").reset_index(drop=True)
        return df_0_augmented

    def augment_rhetq_data(self, df):
        df["tweet"] = df["tweet"].apply(self.augment_rhetq_text)
        df_0_augmented = df.explode("tweet").reset_index(drop=True)
        return df_0_augmented

    # def augment_other_data(self, df):
    #     df["tweet"] = df["tweet"].apply(self.augment_other_text)
    #     df_2_augmented = df.explode("tweet").reset_index(drop=True)
    #     return df_2_augmented

    def remove_sarcasm_overlap(self, df):
        df_new = df.copy()
        aggregated_cols = ["satire", "irony", "overstatement", "understatement", "rhetorical_question"]
        for col in aggregated_cols:
            df_new.loc[(df_new["sarcasm"] == 1) & df_new[col] == 1, "sarcasm"] = 0
        return df_new

    def remove_irony_overlap(self, df):
        df_new = df.copy()
        aggregated_cols = ["satire", "overstatement", "understatement", "rhetorical_question"]
        for col in aggregated_cols:
            df_new.loc[(df_new["irony"] == 1) & df_new[col] == 1, "irony"] = 0
        return df_new

    def load_and_preprocess_training_data(self):
        x_col_types = {
            "tweet": "str",
            # "sarcastic": "int32",
            "sarcasm": "int32",
            "irony": "int32",
            "satire": "int32",
            "understatement": "int32",
            "overstatement": "int32",
            "rhetorical_question": "int32",
        }
        df = (
            pd.read_csv(self.train_data_path)
            .drop(
                columns=[
                    "rephrase",
                    "Unnamed: 0",
                    "sarcastic",
                ]
            )
            .fillna(0)
            .astype(x_col_types)
        )
        df = self.remove_sarcasm_overlap(df)
        df = self.remove_irony_overlap(df)
        df["tweet"] = df["tweet"].apply(
            lambda x: self.remove_contractions(self.remove_urls(self.remove_twitter_handles(x)))
        )
        train_df, val_df = self.split_datasets(df)

        return train_df, val_df

    def augment_data(self, train_df):
        df_sarcasm = train_df[train_df["sarcasm"] == 1]
        df_irony = train_df[train_df["irony"] == 1]
        df_satire = train_df[train_df["satire"] == 1]
        df_overstatement = train_df[train_df["overstatement"] == 1]
        df_understatement = train_df[train_df["understatement"] == 1]
        df_rhet_q = train_df[train_df["rhetorical_question"] == 1]

        df_sarcasm_augmented = self.augment_sarcasm_data(df_sarcasm)
        df_irony_augmented = self.augment_irony_data(df_irony)
        df_satire_augmented = self.augment_satire_data(df_satire)
        df_overstatement_augmented = self.augment_over_data(df_overstatement)
        df_understatement_augmented = self.augment_under_data(df_understatement)
        df_rhetq_augmented = self.augment_rhetq_data(df_rhet_q)

        train_df_augmented = pd.concat(
            [
                df_sarcasm_augmented,
                df_irony_augmented,
                df_satire_augmented,
                df_overstatement_augmented,
                df_understatement_augmented,
                df_rhetq_augmented,
            ]
        )

        return train_df_augmented

    def load_and_preprocess_test_data(self):
        y_col_types = {
            "text": "str",
            # "sarcastic": "int32",
            "sarcasm": "int32",
            "irony": "int32",
            "satire": "int32",
            "understatement": "int32",
            "overstatement": "int32",
            "rhetorical_question": "int32",
        }

        test_df = (
            pd.read_csv(self.test_data_path)
            # .drop(columns=["irony", "satire", "understatement", "overstatement", "rhetorical_question"])
            .fillna(0).astype(y_col_types)
        )

        test_df = self.remove_sarcasm_overlap(test_df)

        test_df["text"] = test_df["text"].apply(
            lambda x: self.remove_contractions(self.remove_urls(self.remove_twitter_handles(x)))
        )

        return test_df

    def compute_weights(self):
        labels_list = [
            "sarcasm",
            "irony",
            "satire",
            "understatement",
            "overstatement",
            "rhetorical_question",
        ]
        y_train = [[record[label] for label in labels_list] for record in self.data_train]
        classes = list(set([item for sublist in y_train for item in sublist]))
        weights = []
        for label in labels_list:
            y_train_label = [record[label] for record in self.data_train]
            weights.append(compute_class_weight(class_weight="balanced", classes=classes, y=y_train_label))
        weights = np.max(np.array(weights), axis=0)
        class_weights = {cls: weight for cls, weight in zip(classes, weights)}
        sample_weights = [max([class_weights[label] for label in record]) for record in y_train]
        self.train_sample_weights = torch.DoubleTensor(sample_weights)
        return class_weights

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_data, self.val_data = self.load_and_preprocess_training_data()
            self.train_data = self.augment_data(self.train_data)
            self.data_train = self.train_data.to_dict("records")
            self.data_val = self.val_data.to_dict("records")
            labels_list = [
                "sarcasm",
                "irony",
                "satire",
                "understatement",
                "overstatement",
                "rhetorical_question",
            ]
            print("Train data class distributions:")
            for label in labels_list:
                print(f"{label}: ", self.train_data[label].value_counts().to_dict())
            print("\nValidation data class distributions:")
            for label in labels_list:
                print(f"{label}: ", self.val_data[label].value_counts().to_dict())
            self.compute_weights()  # compute weights here
            self.train_dataset = SubcategoryDataset(self.data_train, self.tokenizer)
            self.val_dataset = SubcategoryDataset(self.data_val, self.tokenizer)

        if stage == "test":
            self.test_data = self.load_and_preprocess_test_data()
            self.data_test = self.test_data.to_dict("records")
            self.test_dataset = SubcategoryDataset(self.data_test, self.tokenizer)

        if stage == "predict":
            self.predict_dataset = SubcategoryDataset(self.data_test, self.tokenizer)

    def steps_per_epoch(self):
        return len(self.train_dataset)

    def split_datasets(self, df):
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        return train_df, val_df

    def train_dataloader(self):
        sampler = WeightedRandomSampler(self.train_sample_weights, len(self.train_sample_weights))
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=6, sampler=sampler)

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


class ElectraClassifier(pl.LightningModule):
    def __init__(
        self,
        data_module,
        batch_size,
        model_name="google/electra-small-discriminator",
        num_labels=None,
        learning_rate=5e-4,
    ):
        super().__init__()

        serialiazable_hparams = {"num_labels": num_labels, "batch_size": batch_size, "learning_rate": learning_rate}
        self.save_hyperparameters(serialiazable_hparams)
        self.model_name_or_path = model_name
        self.model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.classifier = self.model.classifier
        self.data_module = data_module
        self.num_layers = self.model.config.num_hidden_layers
        self.warmup_steps = 10000
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_labels = num_labels or data_module.num_labels
        self.best_val_performance = -np.inf
        self.epochs_since_best_performance = 0
        self.patience = 2
        self.current_unfreeze_idx = int(len(list(self.model.electra.parameters())))
        self.unfreeze_step = 1
        self.dropout = nn.Dropout(0.15)
        self.f1_macro_scores = []
        self.f1_classes_scores = []
        self.predictions = []
        # self.epoch_train_losses = []

        for param in self.model.electra.parameters():
            param.requires_grad = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_counts = np.array([2200, 2400, 1000, 400, 1450, 3900])
        inverse_counts = 1 / class_counts
        weights = inverse_counts / np.sum(inverse_counts)
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        self.weights = class_weights

        self.loss_fct = nn.CrossEntropyLoss(weight=class_weights)

        # metrics
        self.train_precision = Precision(task="multiclass", num_classes=self.num_labels, average="macro")
        self.train_recall = Recall(task="multiclass", num_classes=self.num_labels, average="macro")
        self.train_f1_macro = F1Score(task="multiclass", num_classes=self.num_labels, average="macro")
        self.train_f1_classes = F1Score(task="multiclass", num_classes=self.num_labels, average=None)

        self.val_precision = Precision(task="multiclass", num_classes=self.num_labels, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=self.num_labels, average="macro")
        self.val_f1_macro = F1Score(task="multiclass", num_classes=self.num_labels, average="macro")
        self.val_f1_classes = F1Score(task="multiclass", num_classes=self.num_labels, average=None)

        self.test_f1_macro = F1Score(task="multiclass", num_classes=self.num_labels, average="macro")
        self.test_f1_classes = F1Score(task="multiclass", num_classes=self.num_labels, average=None)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloder()

    def predict_dataloader(self):
        return self.data_module.predict_dataloader()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        dropout_outputs = self.dropout(hidden_states)
        logits = self.classifier(dropout_outputs)

        if labels is not None:
            loss = self.loss_fct(logits, labels.float())
            loss_labels = torch.argmax(labels, dim=-1)
            return loss, logits
        else:
            return logits

    def on_train_start(self):
        self.train_dataset_len = len(self.train_dataloader().dataset)

    # def on_train_batch_start(self, batch, batch_idx):
    #     if self.global_step == self.warmup_steps:
    #         # unfreeze base layers
    #         self.current_unfreeze_idx = int(len(list(self.model.electra.parameters())) * 0.9)
    #         self.unfreeze_next_layer()

    #         optimizer = self.optimizers()
    #         freeze_idx = int(len(list(self.model.electra.parameters())) * 0.9)

    #         for idx, param_group in enumerate(optimizer.param_groups):
    #             if self.global_step < self.warmup_steps:
    #                 if idx < freeze_idx:
    #                     param_group["lr"] = 0

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        labels = labels.squeeze(1)
        # loss = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # preds = torch.sigmoid(logits) > 0.5  # Convert probabilities to binary predictions
        preds = torch.argmax(logits, dim=-1)
        labels = torch.argmax(labels, dim=-1)
        # preds = torch.argmax(logits, dim=-1)
        # preds = preds.unsqueeze(1)

        f1_macro = self.train_f1_macro(preds, labels)
        f1_classes = self.train_f1_classes(preds, labels)
        prec = self.train_precision(preds, labels)
        rec = self.train_recall(preds, labels)
        # print("Step:", batch_idx, "Loss:", loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_precision", prec, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_recall", rec, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1_macro", f1_macro, on_step=True, on_epoch=True, prog_bar=True)
        for i, score in enumerate(f1_classes):
            self.log(f"Class {i} F1", score, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        labels = labels.squeeze(1)
        # logits = self(input_ids=input_ids, attention_mask=attention_mask)  # Get logits
        # loss = self.loss_fct(logits, labels.float())
        loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # preds = torch.sigmoid(logits) > 0.5
        preds = torch.argmax(logits, dim=-1)
        labels = torch.argmax(labels, dim=-1)
        # preds = torch.argmax(logits, dim=-1)
        # preds = preds.unsqueeze(1)
        # print("shape of preds:", preds.shape)
        # print("shape of labels:", labels.shape)
        f1_macro = self.val_f1_macro(preds, labels)
        f1_classes = self.val_f1_classes(preds, labels)
        prec = self.val_precision(preds, labels).mean()
        rec = self.val_recall(preds, labels).mean()
        # print("Step:", batch_idx, "Loss:", loss.item())
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        self.log("val_precision", prec, on_step=True, on_epoch=True)
        self.log("val_recall", rec, on_step=True, on_epoch=True)
        self.log("val_f1_macro", f1_macro, on_step=True, on_epoch=True)
        for i, score in enumerate(f1_classes):
            self.log(f"Class {i} F1", score, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        labels = labels.squeeze(1)
        # logits = self(input_ids=input_ids, attention_mask=attention_mask)  # Get logits
        # loss = self.loss_fct(logits, labels.float())
        loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # preds = torch.sigmoid(logits) > 0.5
        preds = torch.argmax(logits, dim=-1)
        labels = torch.argmax(labels, dim=-1)
        # preds = torch.argmax(logits, dim=-1)
        # preds = preds.unsqueeze(1)

        f1_macro = self.test_f1_macro(preds, labels)
        f1_classes = self.test_f1_classes(preds, labels)

        self.predictions.append(preds.detach().cpu())
        self.targets.append(labels.detach().cpu())
        self.f1_macro_scores.append(f1_macro.detach().cpu())
        self.f1_classes_scores.append(f1_classes.detach().cpu())

        self.log("test_loss", loss, on_step=True, prog_bar=True)
        # self.log("test_f1", f1_score, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_f1_macro", f1_macro)
        for i, score in enumerate(f1_classes):
            self.log(f"Class {i} F1", score, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_start(self):
        self.predictions = []
        self.targets = []

    def on_test_epoch_end(self):
        # print("predictions on_test_epoch_end:", self.predictions)
        all_preds = torch.cat(self.predictions)
        all_labels = torch.cat(self.targets)
        return {"preds": all_preds, "labels": all_labels}

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        labels = labels.squeeze(1)
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fct(logits, labels.float())

        # preds = torch.sigmoid(logits) > 0.5
        preds = torch.argmax(logits, dim=1)
        f1_macro = self.test_f1_macro(preds, labels)
        f1_classes = self.test_f1_classes(preds, labels)

        self.predictions.append(preds.detach().cpu())
        self.targets.append(labels.detach().cpu())
        self.f1_macro_scores.append(f1_macro.detach().cpu())
        self.f1_classes_scores.append(f1_classes.detach().cpu())

        self.log("test_loss", loss, on_step=True, prog_bar=True)
        self.log("test_f1_macro", f1_macro)
        for i, score in enumerate(f1_classes):
            self.log(f"Class {i} F1", score, on_step=True, on_epoch=True, prog_bar=True)

        return preds, f1_macro

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
        optimizer = RAdam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
        # scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3, verbose=True)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_f1_macro",
        #         "interval": "epoch",
        #         "frequency": 1,
        #         "strict": True,
        #     },
        # }

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
    logger = TensorBoardLogger(save_dir=logdir, name=f"electra_model_noft_v{non_sarc_vnum}")
    metrics_callback = MetricsCallback()
    early_stopping = EarlyStopping("val_loss", patience=5, verbose=True)

    trainer = Trainer(
        max_epochs=1000,
        callbacks=[
            lr_monitor,
            metrics_callback,
            early_stopping,
        ],
        logger=logger,
    )

    trainer.fit(model, data_module)

    trainer.save_checkpoint(new_ckpt_pth)
    # base_model = model.model.base_model
    # base_model.save_pretrained(save_directory)
    return model


def test(model, data_module):
    trainer = Trainer()
    # testing

    test_result = trainer.test(model, data_module)
    print(test_result)
    return model.predictions, model.f1_macro_scores, model.f1_classes_scores


def get_f1_scores(predict_result, macro_f1_scores, classes_f1_scores):
    f1_scores = [f1_score for (_batch_preds, f1_score) in predict_result]
    f1_scores = torch.mean(torch.stack(f1_scores))
    return macro_f1_scores, classes_f1_scores


def tensor_to_native(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_native(v) for v in obj]
    else:
        return obj


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
    sub_data_module = SarcasmSubDataModule(data_path=[sub_data_path_train, sub_data_path_test], batch_size=16)
    model = ElectraClassifier(data_module=sub_data_module, batch_size=sub_data_module.batch_size, num_labels=6)
    model = fit(model, sub_data_module)

    tensorboard_process.terminate()

    sub_data_module.setup("test")
    predictions, macro_f1_scores, classes_f1_scores = test(model, sub_data_module)
    predictions_list = tensor_to_native(predictions)
    f1_scores = get_f1_scores(predictions, macro_f1_scores, classes_f1_scores)
    with open("predictions.json", "w") as f:
        json.dump(predictions_list, f)

    print(f1_scores)

    model = test(model, sub_data_module)


if __name__ == "__main__":
    main()
