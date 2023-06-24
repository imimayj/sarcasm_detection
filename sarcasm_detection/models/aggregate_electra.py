"""
This is the Aggregate ELECTRA Classifier module, which loads in the fine-tuned version
of the ELECTRA Classifier trained on sarcastic data. This is the secondary module
used for assessing the base fine-tuned ELECTRA model on the Semeval 2022 dataset.
This differs from the Custom ELECTRA Classifier in that it assesses the model's
ability to identify the 'sarcasm', 'not_sarcastic', and 'other' classes - where
'other' is the subcategories aggregated into one column
"""

import datetime
import os
import re
import subprocess

import contractions
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from electra_classifier import ElectraClassifier, SarcasmDataModule
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from textattack.augmentation import Augmenter
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.transformations import (
    CompositeTransformation,
    WordSwapExtend,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)
from torch.optim import RAdam
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics import F1Score, Precision, Recall
from transformers import ElectraTokenizer
from transformers.models.electra.modeling_electra import ElectraClassificationHead

# Defining global variables

data_path = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/Sarcasm_Headlines_Dataset_v2.json"
sub_data_path_train = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/train.csv"
sub_data_path_test = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/test.csv"
version_number = 8
sub_version_number = 10
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = f"/workspaces/sarcasm_detection/sarcasm_detection/checkpoints/fine_tuned/sarcasm_detection_finetune_ckpt_v{version_number}_{current_time}.ckpt"
agg_checkpoint_path = f"/workspaces/sarcasm_detection/notebooks/checkpoints/aggregate_trained/subcat_agg_ckpt_v{sub_version_number}_{current_time}.ckpt"
checkpoint_directory = os.path.dirname(checkpoint_path)
logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"
save_directory = (
    f"/workspaces/sarcasm_detection/sarcasm_detection/saved_models/sarcasm_model_v{version_number}_{current_time}"
)


# data module
class SubcategoryDataset(Dataset):
    """
    This is the Dataset class for the module, which gets and sets values to be
    used when preparing the dataset for processing in the model
    """

    def __init__(self, data, tokenizer, max_length=512):
        """
        The constructor method for the class, called upon initialisation of a new
        class. This method sets up essential variables, including the data, tokenizer,
        and max sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # Returning the length of the dataset for other operations throughout
        return len(self.data)

    def __getitem__(self, idx):
        # Identifying the keys in the dataset for labels provided to the
        # Aggregate ELECTRA Classifier module
        text_key = "tweet" if "tweet" in self.data[idx] else "text"
        text = self.data[idx][text_key]

        # Setting the keys to be used as labels
        keys = ["sarcasm", "not_sarcastic", "other"]
        labels = [self.data[idx][key] for key in keys]
        # Encoding the text and labels with the tokenizer, ready to be fed to the model
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
    """
    This is the Datamodule class, which is used to prepare and preprocess the
    Semeval dataset. The data is split into training, validation, testing, and
    prediction datasets depending on the stage passed to the Datamodule class
    by PyTorch Lightning's Trainer class.
    """

    def __init__(self, data_path, batch_size, tokenizer="google/electra-small-discriminator"):
        """
        The constructor method for the class, called upon initialisation of a new
        class. This method sets up essential variables, including the data paths,
        tokenizer, batch size, collate function for data batching, and current
        preprocessing stage
        """
        super().__init__()
        self.train_data_path = data_path[0]
        self.test_data_path = data_path[1]
        self.tokenizer = ElectraTokenizer.from_pretrained(tokenizer)
        self.batch_size = batch_size
        self.collate_fn = self.default_collate_fn

        """
        The below variables are inherited instances of the TextAttack Transformation
        and Augmenter classes. They provide information relating to the transformations
        on a character level applied to each data sample to augment it. The augmenters
        for each class are separate, as functionality is not yet supported to 
        augment classes distinctly based on number of samples to generate (i.e.
        in the instance that 5 generated samples are required for one class, and 
        7 generated samples are required for another). The values were calculated
        based on the existing number of samples in the dataset.
        """
        transformations = CompositeTransformation(
            [WordSwapExtend(), WordSwapRandomCharacterInsertion(), WordSwapRandomCharacterDeletion()]
        )
        constraints = [RepeatModification(), StopwordModification()]
        # The augmenter for sarcasm, which produces 20 samples per data sample,
        # and changes 2% of the words in the sentence.
        self.sarcasm_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=20,
            pct_words_to_swap=0.02,
        )
        # The augmenter for other, which produces 50 samples per data sample,
        # and changes 2% of the words in the sentence.
        self.other_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=50,
            pct_words_to_swap=0.02,
        )

    def remove_twitter_handles(self, text):
        # Function to remove Twitter handles, replacing these with "account_name"
        return re.sub(r"@[\w]+", "account_name", text)

    def remove_urls(self, text):
        # Function to remove URLs, replacing these with "http\S"
        return re.sub(r"http\S+", " ", text)

    def remove_contractions(self, text):
        # Function to remove contractions from the data i.e. 'isn't' -> 'is not'
        return contractions.fix(text)

    def augment_sarcasm_text(self, text):
        # Function to take the 'sarcasm' samples and augment using sarcasm augmenter
        augmented_text = self.sarcasm_augmenter.augment(text)
        return augmented_text

    def augment_other_text(self, text):
        # Function to take the 'other' samples and augment using sarcasm augmenter
        augmented_text = self.other_augmenter.augment(text)
        return augmented_text

    def augment_sarcasm_data(self, df):
        # Function to extract the 'tweet' column text values, applying the
        # respective augment_text functions to each part of the data frame
        # df.explode is applied so that the data frames can later be concatenated
        # to create a new one which includes the new generated data samples
        df["tweet"] = df["tweet"].apply(self.augment_sarcasm_text)
        df_0_augmented = df.explode("tweet")
        return df_0_augmented

    def augment_other_data(self, df):
        # Function to extract the 'tweet' column text values, applying the
        # respective augment_text functions to each part of the data frame
        # df.explode is applied so that the data frames can later be concatenated
        # to create a new one which includes the new generated data samples
        df["tweet"] = df["tweet"].apply(self.augment_other_text)
        df_2_augmented = df.explode("tweet")
        return df_2_augmented

    def remove_sarcasm_overlap(self, df):
        # This function removes the overlap between the 'sarcasm' column
        # and the remaining columns. This sets the value in the sarcasm column
        # to 0 in the columns where sarcasm == 1 and any of the other columns
        # is also == 1.
        df_new = df.copy()
        aggregated_cols = ["satire", "irony", "overstatement", "understatement", "rhetorical_question"]
        for col in aggregated_cols:
            df_new.loc[(df_new["sarcasm"] == 1) & df_new[col] == 1, "sarcasm"] = 0
        return df_new

    def aggregate_cols(self, df):
        # This function aggregates the minority classes into one column named 'other'
        aggregate_cols = ["satire", "irony", "overstatement", "understatement", "rhetorical_question"]
        df["other"] = df[aggregate_cols].any(axis=1).astype("int32")
        df = df.drop(columns=aggregate_cols)
        return df

    def gen_non_sarc(self, df):
        # This function generates the not_sarcastic column by taking all
        # samples where sarcasm & other == 0, in order to invert these values
        # and create positive instances
        df["not_sarcastic"] = ((df["sarcasm"] == 0) & (df["other"] == 0)).astype(int)
        return df

    def prepare_data(self):
        # This is the prepare data function, which prepares the training, validation,
        # and test datasets to be fed to the model.
        aggregate_cols = ["satire", "irony", "overstatement", "understatement", "rhetorical_question"]

        # set column types
        x_col_types = {
            "tweet": "str",
            "sarcasm": "int32",
            "irony": "int32",
            "satire": "int32",
            "understatement": "int32",
            "overstatement": "int32",
            "rhetorical_question": "int32",
        }

        # read in training dataset, dropping columns & filling NaNs, set column types
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

        # remove sarcasm overlap in samples and aggregate the minority columns
        df = self.remove_sarcasm_overlap(df)
        df = self.aggregate_cols(df)

        # apply data preprocessing functions.
        df["tweet"] = df["tweet"].apply(
            lambda x: self.remove_contractions(self.remove_urls(self.remove_twitter_handles(x)))
        )
        # generate not sarcastic column.
        df = self.gen_non_sarc(df)

        # split datasets into training and validation
        train_df, val_df = self.split_datasets(df)
        # split into smaller dfs to later apply data augmentation
        df_sarcasm = train_df[train_df["sarcasm"] == 1]
        df_other = train_df[train_df["other"] == 1]
        df_ns = train_df[train_df["not_sarcastic"] == 1]

        # print sample distribution
        # print("training dataset class distribution before augmentation:")
        # for label in ["sarcasm", "not_sarcastic", "other"]:
        #     train_class_counts = train_df[label].value_counts()
        #     print(f"{label}:")
        #     print(train_class_counts)

        # augment sarcasm and other columns
        df_sarcasm_augmented = self.augment_sarcasm_data(df_sarcasm)
        df_other_augmented = self.augment_other_data(df_other)

        # concatenate resultant dfs into one
        train_df_augmented = pd.concat([df_ns, df_sarcasm_augmented, df_other_augmented])

        # print training and validation dataset distribution after augmentation
        # (applied only to training dataset)
        # print("training dataset class distribution:")
        # for label in ["sarcasm", "not_sarcastic", "other"]:
        #     train_class_counts = train_df_augmented[label].value_counts()
        #     print(f"{label}:")
        #     print(train_class_counts)

        # print("validation dataset class distribution:")
        # for label in ["sarcasm", "not_sarcastic", "other"]:
        #     val_class_counts = val_df[label].value_counts()
        #     print(f"{label}:")
        #     print(val_class_counts)

        # set test dataset column types
        y_col_types = {
            "text": "str",
            "sarcasm": "int32",
            "irony": "int32",
            "satire": "int32",
            "understatement": "int32",
            "overstatement": "int32",
            "rhetorical_question": "int32",
        }

        # Create test dataset by reading in dataset, removing overlap,
        # aggregating columns, generating not_sarcastic column, and applying
        # data preprocessing techniques
        test_df = pd.read_csv(self.test_data_path).fillna(0).astype(y_col_types)
        test_df = self.remove_sarcasm_overlap(test_df)
        test_df = self.aggregate_cols(test_df)
        test_df = self.gen_non_sarc(test_df)
        test_df["text"] = test_df["text"].apply(
            lambda x: self.remove_contractions(self.remove_urls(self.remove_twitter_handles(x)))
        )

        # print out test dataset class distribution

        # print("test dataset class distribution:")
        # for label in ["sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]:
        #     test_class_counts = test_df[label].value_counts()
        #     print(f"{label}:")
        #     print(test_class_counts)

        # Convert train, validation, and test datasets to dicts.
        self.data_train = train_df_augmented.to_dict("records")
        self.data_val = val_df.to_dict("records")
        self.data_test = test_df.to_dict("records")

        # set the label lists
        labels_list = ["sarcasm", "not_sarcastic", "other"]

        # calculate the class weights for oversampling/undersampling
        # Identify class labels in each row in the dataset
        y_train = [[record[label] for label in labels_list] for record in self.data_train]
        # Extract unique class labels in dataset
        classes = list(set([item for sublist in y_train for item in sublist]))
        # Empty list for weights to be used later
        weights = []
        # For each class label, calculate the number of times the label is attached
        # to a data sample. Calculate the class weights for each label
        # (i.e. number of times it appears in dataset)
        for label in labels_list:
            y_train_label = [record[label] for record in self.data_train]
            # compute_class_weights = n_samples / (n_classes * np.bincount(y))
            # 'balanced' uses values of y to adjust weights inversely proportional
            # to class frequencies
            weights.append(compute_class_weight(class_weight="balanced", classes=classes, y=y_train_label))

        # Identify maximum weight for each class from computed weights
        weights = np.max(np.array(weights), axis=0)
        # Dict mapping each class to its weight
        class_weights = {cls: weight for cls, weight in zip(classes, weights)}
        # Obtain maximum class weight among all classes and assign weight to sample
        sample_weights = [max([class_weights[label] for label in record]) for record in y_train]
        # Save sample weights to train_sample_weights as tensor for WeightedRandomSampler
        self.train_sample_weights = torch.DoubleTensor(sample_weights)

    def setup(self, stage: str = None):
        """
        This is the dataloader's setup function. This function takes all of the
        above preprocessing/augmentation steps and wraps it into one function
        in order to be able to produce the correct dataset at each stage based
        on information provided when instantiating the Trainer instance. This is
        so that the datasets don't have to be loaded into memory in their entirety
        every time we just e.g. want to train or test the model.
        """
        if stage == "fit":
            # If the trainer stage is fit, we create the training/validation
            # datasets
            self.train_dataset = SubcategoryDataset(self.data_train, self.tokenizer)
            self.val_dataset = SubcategoryDataset(self.data_val, self.tokenizer)

        if stage == "test":
            # If the stage is test, we create the test dataset
            self.test_dataset = SubcategoryDataset(self.data_test, self.tokenizer)

        if stage == "predict":
            # If the stage is test, we create the predict dataset
            self.predict_dataset = SubcategoryDataset(self.data_test, self.tokenizer)

    def steps_per_epoch(self):
        # define the steps_per_epoch based on length of training dataset
        return len(self.train_dataset)

    def split_datasets(self, df):
        # split dataset function to split the training data into
        # training/validation using an 80/20% split, with random state set for
        # reproducibility
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        return train_df, val_df

    def train_dataloader(self):
        # This instantiates the dataloader for training dataset, which is required
        # for correct PyTorch lightning model implementation to load the correct
        # dataset.
        # The train dataset applies the weights identified above to create a
        # weighted random sampler instance to oversample minority classes.
        sampler = WeightedRandomSampler(self.train_sample_weights, len(self.train_sample_weights))
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=6, sampler=sampler)

    def val_dataloader(self):
        # This instantiates the dataloader for validation dataset, which is required
        # for correct PyTorch lightning model implementation to load the correct
        # dataset.
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=6, shuffle=False)

    def test_dataloader(self):
        # This instantiates the dataloader for test dataset, which is required
        # for correct PyTorch lightning model implementation to load the correct
        # dataset.
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=6, shuffle=False)

    def predict_dataloader(self):
        # This instantiates the dataloader for predict dataset, which is required
        # for correct PyTorch lightning model implementation to load the correct
        # dataset.
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=6, shuffle=False)

    def default_collate_fn(self, batch):
        # This function creates the batch by combining multiple data samples
        # comprising the text, attention mask, and labels.
        input_ids, attention_mask, labels = zip(*batch)
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)
        return input_ids, attention_mask, labels


class CustomElectraClassifier(ElectraClassifier):
    """
    This is the Custom ELECTRA Classifier class. This is the class for outlining
    how the model used will be implemented, providing information to the Trainer so that
    training/validation/testing is executed in the correct way. This inherits
    from the ELECTRA Classifier, to ensure that the fine-tuned model can be used.
    It is almost exactly the same as the Custom ELECTRA Classifier, therefore
    comments are only be provided where there is a difference in implementation
    """

    def __init__(self, data_module=None, batch_size=None, num_labels=None, electra_classifier=None, learning_rate=None):
        super().__init__(data_module=data_module, batch_size=batch_size, num_labels=num_labels)

        if electra_classifier is not None:
            self.model = electra_classifier.model
            self.update_classification_head(num_labels)
        else:
            raise ValueError("An ElectraClassifier must be instantiated")

        serialiazable_hparams = {"num_labels": num_labels, "batch_size": batch_size, "learning_rate": learning_rate}

        self.save_hyperparameters(serialiazable_hparams)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_counts = np.array([8800, 2106, 11400])
        inverse_counts = 1 / class_counts
        weights = inverse_counts / np.sum(inverse_counts)
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)

        # self.model = model_name.from_pretrained(model_name, num_labels=num_labels)
        self.data_module = data_module
        self.num_layers = len(list(self.parameters()))
        self.electra = electra_classifier
        self.warmup_steps = 5000
        self.learning_rate = learning_rate
        self.classifier = self.model.classifier
        self.loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        self.predictions = []
        self.f1_macro_scores = []
        self.f1_classes_scores = []
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.15)

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
        return self.data_module.test_dataloader()

    def predict_dataloader(self):
        return self.data_module.predict_dataloader()

    def forward(self, input_ids, attention_mask, labels=None, num_samples=5):
        """
        The forward pass method is different as we are dealing with a different
        number of labels.
        """
        # pass input ids and attention mas through electra model
        outputs = self.electra(input_ids, attention_mask=attention_mask)

        # check if the model is in training mode
        if self.training:
            # apply dropout and ELECTRA classifier to the outputs for each sample
            logits_list = []
            for _ in range(num_samples):
                dropout_outputs = self.dropout(outputs[0])
                logits = self.classifier(dropout_outputs)
                # append the result to the logits list
                logits_list.append(logits)

            # stack the logits together and compute mean across the logits
            logits = torch.stack(logits_list).mean(0)

        # if not in training, just apply the classifier to outputs without dropout
        else:
            logits = self.classifier(outputs[0])

        # check whether there are labels
        if labels is not None:
            # compute loss based on CrossEntropy loss
            class_idx_labels = torch.argmax(labels, dim=-1)
            loss = self.loss_fct(logits, class_idx_labels)
            # return loss and logits if there are labels
            return loss, logits
        # or just returning logits if labels is none
        else:
            return logits

    def on_train_start(self):
        # The same as in the Custom ELECTRA Classifier
        self.train_dataset_len = len(self.train_dataloader().dataset)

    def on_train_batch_start(self, batch, batch_idx):
        # The same as in the Custom ELECTRA Classifier
        pass

    def on_validation_epoch_end(self):
        # The same as in the Custom ELECTRA Classifier
        pass

    def unfreeze_next_layer(self):
        # The same as in the Custom ELECTRA Classifier
        pass

    def training_step(self, batch, batch_idx):
        # The same as in the Custom ELECTRA Classifier
        input_ids, attention_mask, labels = batch
        labels = labels.squeeze(1)
        loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = torch.argmax(logits, dim=-1)
        labels = torch.argmax(labels, dim=-1)

        f1_macro = self.train_f1_macro(preds, labels)
        f1_classes = self.train_f1_classes(preds, labels)
        prec = self.train_precision(preds, labels)
        rec = self.train_recall(preds, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_precision", prec, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_recall", rec, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1_macro", f1_macro, on_step=True, on_epoch=True, prog_bar=True)
        for i, score in enumerate(f1_classes):
            self.log(f"Class {i} F1", score, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # The same as in the Custom ELECTRA Classifier
        input_ids, attention_mask, labels = batch
        labels = labels.squeeze(1)
        loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = torch.argmax(logits, dim=-1)
        labels = torch.argmax(labels, dim=-1)

        f1_macro = self.val_f1_macro(preds, labels)
        f1_classes = self.val_f1_classes(preds, labels)
        prec = self.val_precision(preds, labels).mean()
        rec = self.val_recall(preds, labels).mean()

        self.log("val_loss", loss, on_step=True, prog_bar=True)
        self.log("val_precision", prec, on_step=True, on_epoch=True)
        self.log("val_recall", rec, on_step=True, on_epoch=True)
        self.log("val_f1_macro", f1_macro, on_step=True, on_epoch=True)
        for i, score in enumerate(f1_classes):
            self.log(f"Class {i} F1", score, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # The same as in the Custom ELECTRA Classifier
        input_ids, attention_mask, labels = batch
        labels = labels.squeeze(1)
        loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = torch.argmax(logits, dim=-1)
        labels = torch.argmax(labels, dim=-1)

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
        # The same as in the Custom ELECTRA Classifier
        self.predictions = []
        self.targets = []

    def on_test_epoch_end(self):
        # The same as in the Custom ELECTRA Classifier
        """
        PTL callback called at the end of a test epoch, for computing and logging
        metrics or calculations once test step is complete.
        """
        # Concatenate all predictions & true labels gathered during test step
        # into two tensors.
        all_preds = torch.cat(self.predictions)
        all_labels = torch.cat(self.targets)

        # Convert these tensors to a numpy array
        all_preds_np = all_preds.numpy()
        all_labels_np = all_labels.numpy()

        # instantiate confusion matrix for evaluating the model's precision &
        # recall manually
        cm = confusion_matrix(all_labels_np, all_preds_np, labels=list(range(self.num_labels)))

        # Calculate the true positives, true negatives, false positives, and
        # false negatives for each class in the range of the number of labels
        for i in range(self.num_labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (fp + fn + tp)
            print(f"Class {i}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

        return {"preds": all_preds, "labels": all_labels}

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        labels = labels.squeeze(1)
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fct(logits, labels.float())

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
        """
        The configure optimizer function for Aggregate ELECTRA is different.
        Implementation experimented with a OneCycleLR learning rate scheduler,
        which varies the learning rate during each cycle. This was beneficial for
        performance in Aggregate-ELECTRA, but led to worse performance with the
        base Custom ELECTRA Classifier.
        """

        # Configure RAdam optimizer with weight decay
        optimizer = RAdam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        # calculate the number of total training steps using the length of the
        # training dataset and the max epochs variable assigned when instantiating
        # the model using Trainer()
        num_training_steps = len(self.train_dataloader()) * self.trainer.max_epochs

        # OneCycleLR - varies the learning rate by 2* the instantiated learning rate
        # and an unknown minimum value during each cycle of training
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2 * self.hparams.learning_rate,
            total_steps=num_training_steps,
            # Using a linear learning rate decrease
            anneal_strategy="linear",
        )

        # return a dictionary of the optimizer, scheduler, and the monitor (validation loss)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class MetricsCallback(Callback):
    """
    This class is the same as the ELECTRA Classifier and Custom ELECTRA CLassifier.
    It is used for recording various metrics at different points during training
    and validation steps.
    """

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
    # The same function as in Custom ELECTRA Classifier
    checkpoints = [
        file for file in os.listdir(checkpoint_directory) if file.startswith(version_prefix) and file.endswith(".ckpt")
    ]
    return (
        max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_directory, x))) if checkpoints else None
    )


def load_model(transfer_data_module, saved_data_module):
    # The same load function as in the Custom ELECTRA Classifier
    latest_checkpoint = find_latest_checkpoint()

    if latest_checkpoint:
        full_checkpoint_path = os.path.join(checkpoint_directory, latest_checkpoint)

        loaded_model = ElectraClassifier.load_from_checkpoint(full_checkpoint_path, data_module=saved_data_module)

        transfer_model = CustomElectraClassifier(
            electra_classifier=loaded_model,
            data_module=transfer_data_module,
            num_labels=3,
            batch_size=transfer_data_module.batch_size,
            learning_rate=5e-4,
        )

        transfer_model.model.electra.load_state_dict(loaded_model.model.electra.state_dict())

        return transfer_model

    else:
        print("No model checkpoint found")


def fit(model, data_module):
    # The same as the Custom ELECTRA Classifier, though the max epochs is higher
    # for the learning rate scheduler max steps to be very large
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=logdir, name=f"esubcat_model_v{sub_version_number}")
    metrics_callback = MetricsCallback()
    early_stopping = EarlyStopping("val_loss", patience=5, verbose=True, mode="min")

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

    trainer.save_checkpoint(agg_checkpoint_path)
    return model


def test(model, data_module):
    # The same as the Custom ELECTRA Classifier test function
    trainer = Trainer()

    test_result = trainer.test(model, data_module)
    return model.predictions, model.f1_macro_scores, model.f1_classes_scores


def get_f1_scores(predictions, macro_f1_scores, classes_f1_scores):
    # The same as the Custom ELECTRA Classifier
    macro_f1_scores = torch.mean(torch.stack(macro_f1_scores))
    classes_f1_scores = torch.mean(torch.stack(classes_f1_scores))
    return macro_f1_scores, classes_f1_scores


def predict(model, data_module):
    # The same as the Custom ELECTRA Classifier (not currently used)
    trainer = Trainer()

    predict_result = trainer.predict(model, data_module)

    print(predict_result)

    f1_scores = get_f1_scores(predict_result)
    print(f"f1 score for predictions: {torch.IntTensor.item(f1_scores)}")


def launch_tensorboard(logdir):
    # The same as the Custom ELECTRA CLassifier
    command = f"tensorboard --logdir={logdir}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return process


def tensor_to_native(obj):
    # The same as the Custom ELECTRA Classifier
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_native(v) for v in obj]
    else:
        return obj


def main():
    # The same as the Custom ELECTRA Classifier
    saved_data_module = SarcasmDataModule(data_path=data_path, batch_size=16)
    sub_data_module = SarcasmSubDataModule(data_path=[sub_data_path_train, sub_data_path_test], batch_size=16)
    transfer_model = load_model(saved_data_module=saved_data_module, transfer_data_module=sub_data_module)
    if transfer_model is not None:
        logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"
        tensorboard_process = launch_tensorboard(logdir)
        transfer_model = fit(transfer_model, sub_data_module)
        tensorboard_process.terminate()

        sub_data_module.setup("test")
        predictions, macro_f1_scores, classes_f1_scores = test(transfer_model, sub_data_module)
        predictions_list = tensor_to_native(predictions)
        f1_scores = get_f1_scores(predictions, macro_f1_scores, classes_f1_scores)
        # output_dict = {"predictions": predictions_list, "true_labels": true_labels}
        # with open("predictions.json", "w") as f:
        #     json.dump(output_dict, f)

        print(f1_scores)

    else:
        print("failed to load the transfer model")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
