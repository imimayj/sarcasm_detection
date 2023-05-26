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
from electra_classifier import ElectraClassifier, SarcasmDataModule, SarcasmDataset
from finetuning_scheduler import FinetuningScheduler
from nltk.corpus import stopwords, wordnet
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from textattack.augmentation import Augmenter, CharSwapAugmenter, DeletionAugmenter, EasyDataAugmenter, WordNetAugmenter
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.transformations import (
    BackTranslation,
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapExtend,
    WordSwapQWERTY,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)

# from textattack.transformations.sentence_transformations import BackTranslation
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler, random_split
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from transformers import (  # cosine_schedule_with_warmup,; get_linear_schedule_with_warmup,
    AdamW,
    AutoTokenizer,
    DataCollatorWithPadding,
    ElectraConfig,
    ElectraForSequenceClassification,
    ElectraModel,
    ElectraTokenizer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.electra.modeling_electra import ElectraClassificationHead

data_path = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/Sarcasm_Headlines_Dataset_v2.json"
sub_data_path_train = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/train.csv"
sub_data_path_test = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/test.csv"
version_number = 6
sub_version_number = 9
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = f"/workspaces/sarcasm_detection/sarcasm_detection/checkpoints/sarcasm_detection_finetune_ckpt_v{version_number}_{current_time}.ckpt"
sub_checkpoint_path = f"/workspaces/sarcasm_detection/sarcasm_detection/checkpoints/subcat_finetune_ckpt_v{sub_version_number}_{current_time}.ckpt"
checkpoint_directory = os.path.dirname(checkpoint_path)
logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"
save_directory = (
    f"/workspaces/sarcasm_detection/sarcasm_detection/saved_models/sarcasm_model_v{version_number}_{current_time}"
)


# data module
class SubcategoryDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_key = "tweet" if "tweet" in self.data[idx] else "text"
        text = self.data[idx][text_key]

        # text = self.data[idx]["tweet"]
        # keys = ["sarcasm", "not_sarcastic", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]
        keys = ["sarcasm", "not_sarcastic", "other"]
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
    def __init__(self, data_path, batch_size, tokenizer="google/electra-small-discriminator"):
        super().__init__()
        self.train_data_path = data_path[0]
        self.test_data_path = data_path[1]
        self.tokenizer = ElectraTokenizer.from_pretrained(tokenizer)
        self.batch_size = batch_size
        self.collate_fn = self.default_collate_fn
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
        self.other_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=10,
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

    def augment_other_text(self, text):
        augmented_text = self.other_augmenter.augment(text)
        return augmented_text

    # def augment_data(self, df, augmenter):
    #     aug_texts = []
    #     aug_labels = []

    #     for index, row in df.iterrows():
    #         # Applying data augmentation method
    #         new_texts = augmenter.augment(row["tweet"])

    #         # Extend augmented texts and labels
    #         aug_texts.extend(new_texts)
    #         aug_labels.extend([row["sarcasm"]] * len(new_texts))

    #     # Creating new dataframe with augmented data
    #     df_augmented = pd.DataFrame(
    #         {
    #             "tweet": aug_texts,
    #             "sarcasm": aug_labels,
    #         }
    #     )

    #     return df_augmented

    def augment_sarcasm_data(self, df):
        df["tweet"] = df["tweet"].apply(self.augment_sarcasm_text)
        df_0_augmented = df.explode("tweet")
        return df_0_augmented

    def augment_other_data(self, df):
        df["tweet"] = df["tweet"].apply(self.augment_other_text)
        df_2_augmented = df.explode("tweet")
        return df_2_augmented

    def remove_sarcasm_overlap(self, df):
        df_new = df.copy()
        aggregated_cols = ["satire", "irony", "overstatement", "understatement", "rhetorical_question"]
        for col in aggregated_cols:
            df_new.loc[(df_new["sarcasm"] == 1) & df_new[col] == 1, "sarcasm"] = 0
        return df_new

    def aggregate_cols(self, df):
        aggregate_cols = ["satire", "irony", "overstatement", "understatement", "rhetorical_question"]
        df["other"] = df[aggregate_cols].any(axis=1).astype("int32")
        df = df.drop(columns=aggregate_cols)
        return df

    def gen_non_sarc(self, df):
        df["not_sarcastic"] = ((df["sarcasm"] == 0) & (df["other"] == 0)).astype(int)
        return df

    def prepare_data(self):
        aggregate_cols = ["satire", "irony", "overstatement", "understatement", "rhetorical_question"]

        x_col_types = {
            "tweet": "str",
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
        df = self.aggregate_cols(df)

        df["tweet"] = df["tweet"].apply(
            lambda x: self.remove_contractions(self.remove_urls(self.remove_twitter_handles(x)))
        )
        df = self.gen_non_sarc(df)

        train_df, val_df = self.split_datasets(df)
        df_sarcasm = train_df[train_df["sarcasm"] == 1]
        df_other = train_df[train_df["other"] == 1]
        df_ns = train_df[train_df["not_sarcastic"] == 1]

        df_sarcasm_augmented = self.augment_sarcasm_data(df_sarcasm)
        df_other_augmented = self.augment_other_data(df_other)

        train_df_augmented = pd.concat([df_ns, df_sarcasm_augmented, df_other_augmented])

        # augmented_other = self.augment_data(train_df[train_df["other"] == 1], self.other_augmenter)
        # print("sarcasm value counts after other augmentation:")
        # print(train_df_augmented_other["sarcasm"].value_counts())

        # # Apply the augmentation for 'sarcasm' class
        # train_df_augmented_sarcasm = self.augment_data(train_df[train_df["sarcasm"] == 1], self.sarcasm_augmenter)train_df_

        # Concatenate original and augmented dataframes
        # train_df_augmented = pd.concat([train_df, train_df_augmented_other, train_df_augmented_sarcasm])

        print("training dataset class distribution:")
        for label in ["sarcasm", "not_sarcastic", "other"]:
            train_class_counts = train_df_augmented[label].value_counts()
            print(f"{label}:")
            print(train_class_counts)

        print("validation dataset class distribution:")
        for label in ["sarcasm", "not_sarcastic", "other"]:
            val_class_counts = val_df[label].value_counts()
            print(f"{label}:")
            print(val_class_counts)

        y_col_types = {
            "text": "str",
            "sarcasm": "int32",
            "irony": "int32",
            "satire": "int32",
            "understatement": "int32",
            "overstatement": "int32",
            "rhetorical_question": "int32",
        }

        test_df = pd.read_csv(self.test_data_path).fillna(0).astype(y_col_types)
        test_df = self.remove_sarcasm_overlap(test_df)
        test_df = self.aggregate_cols(test_df)
        test_df = self.gen_non_sarc(test_df)
        test_df["text"] = test_df["text"].apply(
            lambda x: self.remove_contractions(self.remove_urls(self.remove_twitter_handles(x)))
        )
        # print("test dataset class distribution:")
        # for label in ["sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]:
        #     test_class_counts = test_df[label].value_counts()
        #     print(f"{label}:")
        #     print(test_class_counts)

        # print(f"test df length: {len(train_df)}")
        # print(f"Validation DataFrame length: {len(val_df)}")
        # print(f"Test DataFrame length: {len(test_df)}")
        # print(f"total df len: {len(train_df+val_df+test_df)}")
        # print(len(df))

        self.data_train = train_df_augmented.to_dict("records")
        self.data_val = val_df.to_dict("records")
        self.data_test = test_df.to_dict("records")

        labels_list = ["sarcasm", "not_sarcastic", "other"]

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


class CustomElectraClassifier(ElectraClassifier):
    def __init__(self, data_module=None, batch_size=None, num_labels=None, electra_classifier=None, learning_rate=None):
        super().__init__(data_module=data_module, batch_size=batch_size, num_labels=num_labels)

        if electra_classifier is not None:
            self.model = electra_classifier.model
            self.update_classification_head(num_labels)
        else:
            raise ValueError("An ElectraClassifier must be instantiated")

        serialiazable_hparams = {"num_labels": num_labels, "batch_size": batch_size, "learning_rate": learning_rate}

        self.save_hyperparameters(serialiazable_hparams)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # class_counts = np.array([545, 2229, 124, 20, 7, 29, 77])
        # inverse_counts = 1 / class_counts
        # weights = inverse_counts / np.sum(inverse_counts)
        # class_weights = torch.tensor(weights, dtype=torch.float).to(device)

        # self.model = model_name.from_pretrained(model_name, num_labels=num_labels)
        self.data_module = data_module
        self.num_layers = len(list(self.parameters()))
        self.electra = electra_classifier
        self.warmup_steps = None
        self.learning_rate = learning_rate
        self.classifier = self.model.classifier
        # self.loss_fct = nn.BCEWithLogitsLoss()
        # self.loss_fct = nn.CrossEntropyLoss()
        self.predictions = []
        self.f1_macro_scores = []
        self.f1_classes_scores = []
        self.num_labels = num_labels
        # self.dropout = nn.Dropout(0.15)
        # self.weights = class_weights
        self.loss_fct = nn.CrossEntropyLoss()

        # self.additional_layer_1 = electra_classifier.additional_layer_1
        # self.activation = electra_classifier.activation
        # self.classifier = nn.Linear(64, num_labels)
        # self.dropout = nn.Dropout(0)
        # self.predictions = []
        # self.targets = []

        for param in self.electra.base_model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

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

        # for adding smaller networks on top
        # self.dropout = nn.Dropout(0.1)
        # self.fc1 = nn.Linear(self.electra.config.hidden_size, 128)
        # self.fc2 = nn.Linear(128, num_labels)

    @property
    def electra(self):
        return self.model.electra

    def update_classification_head(self, num_labels):
        config = self.model.config
        config.num_labels = num_labels
        new_classifier = ElectraClassificationHead(config)
        self.model.classifier = new_classifier

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloder()

    def predict_dataloader(self):
        return self.data_module.predict_dataloader()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.electra(input_ids, attention_mask=attention_mask)
        # dropout_outputs = self.dropout(outputs[0])
        # logits = self.classifier(dropout_outputs)
        logits = self.classifier(outputs[0])
        # logits = self.classifier(outputs[0])
        # print(f"logits shape: {logits.shape}")

        # if labels is not None:
        #     loss = self.loss_fct(logits, labels.float())
        #     # loss = self.loss_fct(logits.view(-1, self.num_labels), labels)
        #     return loss
        # else:
        #     return logits

        if labels is not None:
            class_idx_labels = torch.argmax(labels, dim=-1)
            loss = self.loss_fct(logits, class_idx_labels)
            # loss_labels = torch.argmax(labels, dim=-1)
            # print("Logits shape:", logits.shape)
            # print("Labels shape:", labels.shape)
            # loss = self.loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

    # def update_classification_head(self, num_labels):
    #     self.classifier = nn.Linear(64, num_labels)

    def on_train_start(self):
        self.train_dataset_len = len(self.train_dataloader().dataset)

    def on_train_batch_start(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        pass

    def unfreeze_next_layer(self):
        pass

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
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # return optimizer
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1_macro",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }

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


def load_model(transfer_data_module, saved_data_module):  # also saved_data_module
    latest_checkpoint = find_latest_checkpoint()

    if latest_checkpoint:
        full_checkpoint_path = os.path.join(checkpoint_directory, latest_checkpoint)

        # load model
        loaded_model = ElectraClassifier.load_from_checkpoint(full_checkpoint_path, data_module=saved_data_module)

        # new model with different classification heads
        transfer_model = CustomElectraClassifier(
            electra_classifier=loaded_model,
            data_module=transfer_data_module,
            num_labels=3,
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
    early_stopping = EarlyStopping("val_loss", patience=5, verbose=True, mode="min")

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
    return model


def test(model, data_module):
    trainer = Trainer()
    # testing

    test_result = trainer.test(model, data_module)
    # print(test_result)
    return model.predictions, model.f1_macro_scores, model.f1_classes_scores


def get_f1_scores(predictions, macro_f1_scores, classes_f1_scores):
    # f1_scores = [f1_score for (_batch_preds, f1_score) in predict_result]
    macro_f1_scores = torch.mean(torch.stack(macro_f1_scores))
    classes_f1_scores = torch.mean(torch.stack(classes_f1_scores))
    return macro_f1_scores, classes_f1_scores


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


def tensor_to_native(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_native(v) for v in obj]
    else:
        return obj


def main():
    saved_data_module = SarcasmDataModule(data_path=data_path, batch_size=16)
    sub_data_module = SarcasmSubDataModule(data_path=[sub_data_path_train, sub_data_path_test], batch_size=16)
    transfer_model = load_model(saved_data_module=saved_data_module, transfer_data_module=sub_data_module)
    if transfer_model is not None:
        logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"
        tensorboard_process = launch_tensorboard(logdir)
        transfer_model = fit(transfer_model, sub_data_module)
        tensorboard_process.terminate()
        predictions, macro_f1_scores, classes_f1_scores = test(transfer_model, sub_data_module)
        predictions_list = tensor_to_native(predictions)
        f1_scores = get_f1_scores(predictions, macro_f1_scores, classes_f1_scores)
        with open("predictions.json", "w") as f:
            json.dump(predictions_list, f)

        print(f1_scores)

    else:
        print("failed to load the transfer model")


if __name__ == "__main__":
    main()
