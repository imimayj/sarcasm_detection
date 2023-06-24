"""
This is the Custom ELECTRA Classifier module, which loads in the fine-tuned version
of the ELECTRA Classifier trained on sarcastic data. This is the primary module
used for assessing the base fine-tuned ELECTRA model on the Semeval 2022 dataset.
"""


import datetime
import json
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
sub_version_number = 9
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = f"/workspaces/sarcasm_detection/sarcasm_detection/checkpoints/fine_tuned/sarcasm_detection_finetune_ckpt_v{version_number}_{current_time}.ckpt"
sub_checkpoint_path = f"/workspaces/sarcasm_detection/notebooks/checkpoints/custom_trained/subcat_finetune_ckpt_v{sub_version_number}_{current_time}.ckpt"
checkpoint_directory = os.path.dirname(checkpoint_path)
logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"
save_directory = f"/workspaces/sarcasm_detection/sarcasm_detection/saved_models/custom_electra_model_v{version_number}_{current_time}"


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
        # Custom ELECTRA Classifier module
        text_key = "tweet" if "tweet" in self.data[idx] else "text"
        text = self.data[idx][text_key]

        # Setting the keys to be used as labels
        keys = ["sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]
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

    def __init__(self, data_path, batch_size, tokenizer="google/electra-small-discriminator", stage=None):
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
        self.stage = stage

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

        # The augmenter for sarcasm, which produces 5 samples per data sample,
        # and changes 2% of the words in the sentence.
        self.sarcasm_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=5,
            pct_words_to_swap=0.02,
        )

        # The augmenter for satire, which produces 50 samples per data sample,
        # and changes 2.5% of the words in the sentence.
        self.satire_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=50,
            pct_words_to_swap=0.025,
        )

        # The augmenter for irony, which produces 25 samples per data sample,
        # and changes 2% of the words in the sentence.
        self.irony_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=25,
            pct_words_to_swap=0.025,
        )

        # The augmenter for understatement, which produces 60 samples per data sample,
        # and changes 2% of the words in the sentence.
        self.under_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=60,
            pct_words_to_swap=0.03,
        )

        # The augmenter for overstatement, which produces 50 samples per data sample,
        # and changes 2% of the words in the sentence.
        self.over_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=50,
            pct_words_to_swap=0.03,
        )

        # The augmenter for rhetorical question, which produces 50 samples per
        # data sample, and changes 2% of the words in the sentence.
        self.rhetq_augmenter = Augmenter(
            transformation=transformations,
            constraints=constraints,
            transformations_per_example=40,
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

    def augment_irony_text(self, text):
        # Function to take the 'irony' samples and augment using irony augmenter
        augmented_text = self.irony_augmenter.augment(text)
        return augmented_text

    def augment_satire_text(self, text):
        # Function to take the 'satire' samples and augment using satire augmenter
        augmented_text = self.satire_augmenter.augment(text)
        return augmented_text

    def augment_under_text(self, text):
        # Function to take the 'understatement' samples and augment using
        # understatement augmenter
        augmented_text = self.under_augmenter.augment(text)
        return augmented_text

    def augment_over_text(self, text):
        # Function to take the 'overstatement' samples and augment using
        # overstatement augmenter
        augmented_text = self.over_augmenter.augment(text)
        return augmented_text

    def augment_rhetq_text(self, text):
        # Function to take the 'rhetorical question' samples and augment using
        # rhetorical question augmenter
        augmented_text = self.rhetq_augmenter.augment(text)
        return augmented_text

    def augment_sarcasm_data(self, df):
        # Function to extract the 'tweet' column text values, applying the
        # respective augment_text functions to each part of the data frame
        # df.explode is applied so that the data frames can later be concatenated
        # to create a new one which includes the new generated data samples
        df["tweet"] = df["tweet"].apply(self.augment_sarcasm_text)
        df_0_augmented = df.explode("tweet").reset_index(drop=True)
        return df_0_augmented

    def augment_irony_data(self, df):
        # Function to extract the 'tweet' column text values, applying the
        # respective augment_text functions to each part of the data frame
        # df.explode is applied so that the data frames can later be concatenated
        # to create a new one which includes the new generated data samples
        df["tweet"] = df["tweet"].apply(self.augment_irony_text)
        df_0_augmented = df.explode("tweet").reset_index(drop=True)
        return df_0_augmented

    def augment_satire_data(self, df):
        # Function to extract the 'tweet' column text values, applying the
        # respective augment_text functions to each part of the data frame
        # df.explode is applied so that the data frames can later be concatenated
        # to create a new one which includes the new generated data samples
        df["tweet"] = df["tweet"].apply(self.augment_satire_text)
        df_0_augmented = df.explode("tweet").reset_index(drop=True)
        return df_0_augmented

    def augment_under_data(self, df):
        # Function to extract the 'tweet' column text values, applying the
        # respective augment_text functions to each part of the data frame
        # df.explode is applied so that the data frames can later be concatenated
        # to create a new one which includes the new generated data samples
        df["tweet"] = df["tweet"].apply(self.augment_under_text)
        df_0_augmented = df.explode("tweet").reset_index(drop=True)
        return df_0_augmented

    def augment_over_data(self, df):
        # Function to extract the 'tweet' column text values, applying the
        # respective augment_text functions to each part of the data frame
        # df.explode is applied so that the data frames can later be concatenated
        # to create a new one which includes the new generated data samples
        df["tweet"] = df["tweet"].apply(self.augment_over_text)
        df_0_augmented = df.explode("tweet").reset_index(drop=True)
        return df_0_augmented

    def augment_rhetq_data(self, df):
        # Function to extract the 'tweet' column text values, applying the
        # respective augment_text functions to each part of the data frame
        # df.explode is applied so that the data frames can later be concatenated
        # to create a new one which includes the new generated data samples
        df["tweet"] = df["tweet"].apply(self.augment_rhetq_text)
        df_0_augmented = df.explode("tweet").reset_index(drop=True)
        return df_0_augmented

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

    def remove_irony_overlap(self, df):
        # This function removes the overlap between the 'irony' column
        # and the remaining columns (not sarcasm as this has already been removed).
        # This sets the value in the irony column to 0 in the columns where
        # irony == 1 and any of the other columns is also == 1.
        df_new = df.copy()
        aggregated_cols = ["satire", "overstatement", "understatement", "rhetorical_question"]
        for col in aggregated_cols:
            df_new.loc[(df_new["irony"] == 1) & df_new[col] == 1, "irony"] = 0
        return df_new

    def remove_rhetq_overlap(self, df):
        # This function removes the overlap between the 'rhetorical question' column
        # and the remaining columns (not sarcasm, irony as these have already
        # been removed). This sets the value in the irony column to 0 in the
        # columns where irony == 1 and any of the other columns is also == 1.
        df_new = df.copy()
        aggregated_cols = ["understatement"]
        for col in aggregated_cols:
            df_new.loc[(df_new["rhetorical_question"] == 1) & (df_new[col] == 1), "rhetorical_question"] = 0
        return df_new

    # unused function for creating a not_sarcastic class, which led to inferior results
    # def gen_non_sarc(self, df):
    #     not_sarcastic_cond = (
    #         (df["sarcasm"] == 0)
    #         & (df["satire"] == 0)
    #         & (df["irony"] == 0)
    #         & (df["overstatement"] == 0)
    #         & (df["understatement"] == 0)
    #         & (df["rhetorical_question"] == 0)
    #     )

    #     print(f"Total 'not_sarcastic' samples: {not_sarcastic_cond.sum()}")

    #     df["not_sarcastic"] = not_sarcastic_cond.astype(int)
    #     return df

    def load_and_preprocess_training_data(self):
        """
        This function is used for loading and preprocessing the training dataset.
        It takes in the dataset converting it to a dataframe, dropping
        unnecessary columns, setting the labels for columns and the column types,
        applies the above functions for removal of overlap between columns,
        splits the datasets into training/validation datasets, and applies
        data augmentation techniques ONLY to the training dataset to ensure that
        validation data correct to the samples is used.

        NB: there are also commented out print statements for verifying that
        each step is being applied correctly, and to identify changes in sample
        distribution
        """
        # Setting column types
        x_col_types = {
            "tweet": "str",
            "sarcasm": "int32",
            "irony": "int32",
            "satire": "int32",
            "understatement": "int32",
            "overstatement": "int32",
            "rhetorical_question": "int32",
        }

        # Basic preprocessing/reading fo the dataset, dropping columns, filling
        # NAN values etc
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

        # Setting the label list to identify class distributions later
        labels_list = [
            "sarcasm",
            "irony",
            "satire",
            "understatement",
            "overstatement",
            "rhetorical_question",
        ]
        # Remove non-sarcastic samples from the dataset
        df = df.loc[~(df[labels_list] == 0).all(axis=1)]

        # Print statements to verify that each stage is being applied correctly
        # print("dataset class distributions before sarcasm overlap removal:")
        # for label in labels_list:
        #     print(f"{label}: ", df[label].value_counts().to_dict())
        # df = self.remove_sarcasm_overlap(df)
        # print("dataset class distributions after sarcasm removal and before irony removal:")
        # for label in labels_list:
        #     print(f"{label}: ", df[label].value_counts().to_dict())
        # df = self.remove_irony_overlap(df)
        # print("dataset class distribution after sarcasm and irony removal:")
        # for label in labels_list:
        #     print(f"{label}: ", df[label].value_counts().to_dict())
        # df = self.remove_rhetq_overlap(df)
        # print("dataset class distribution after sarcasm, irony, and rhetq removal:")
        # for label in labels_list:
        #     print(f"{label}: ", df[label].value_counts().to_dict())

        # Applying the preprocessing functions delineated above
        df["tweet"] = df["tweet"].apply(
            lambda x: self.remove_contractions(self.remove_urls(self.remove_twitter_handles(x)))
        )

        # Splitting the datasets using split_datasets function (below)
        train_df, val_df = self.split_datasets(df)

        # Print statements for verifying that augmentation is being correctly applied
        # print("train dataset class distribution before augmentation:")
        # for label in labels_list:
        #     print(f"{label}: ", train_df[label].value_counts().to_dict())
        # print("val dataset class distribution before augmentation:")
        # for label in labels_list:
        #     print(f"{label}: ", val_df[label].value_counts().to_dict())

        # return datasets for later use
        return train_df, val_df

    def augment_data(self, train_df):
        """
        This function is for applying data augmentation techniques.
        First, it identifies the correct df subsets to apply the techniques -
        distinguished by each label to correctly apply the relevant function
        using the correct augmenter. Augmentation techniques using the above
        functions are then applied to each subset, then the new dataframes are
        subsequently concatenated together to prevent issues with the dataset.
        """

        # Identify subsets of df
        df_sarcasm = train_df[train_df["sarcasm"] == 1]
        df_irony = train_df[train_df["irony"] == 1]
        df_satire = train_df[train_df["satire"] == 1]
        df_overstatement = train_df[train_df["overstatement"] == 1]
        df_understatement = train_df[train_df["understatement"] == 1]
        df_rhet_q = train_df[train_df["rhetorical_question"] == 1]

        # Apply relevant data augmentation functions
        df_sarcasm_augmented = self.augment_sarcasm_data(df_sarcasm)
        df_irony_augmented = self.augment_irony_data(df_irony)
        df_satire_augmented = self.augment_satire_data(df_satire)
        df_overstatement_augmented = self.augment_over_data(df_overstatement)
        df_understatement_augmented = self.augment_under_data(df_understatement)
        df_rhetq_augmented = self.augment_rhetq_data(df_rhet_q)

        # Concatenate the resultant dataframes
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

        # return the new df for use in model
        return train_df_augmented

    def load_and_preprocess_test_data(self):
        """
        This function is for loading and preprocessing the test dataset, which
        is slightly different in formatting than the training dataset. This
        function reads the dataset and applies basic data integrity processes,
        removes overlap between the classes for consistency in line with the
        training dataset, returning it to be used later.
        """

        # Set the column types
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

        # Apply basic data integrity measures, filling NaNs.
        test_df = pd.read_csv(self.test_data_path).fillna(0).astype(y_col_types)

        # Set the label list to print class sample distributions
        labels_list = [
            "sarcasm",
            "irony",
            "satire",
            "understatement",
            "overstatement",
            "rhetorical_question",
        ]
        test_df = test_df.loc[~(test_df[labels_list] == 0).all(axis=1)]

        # Print class samples before overlap removal
        # print("test dataset class distribution:")
        # for label in labels_list:
        #     print(f"{label}: ", test_df[label].value_counts().to_dict())

        # Apply overlap removal functions
        test_df = self.remove_sarcasm_overlap(test_df)
        test_df = self.remove_irony_overlap(test_df)
        test_df = self.remove_rhetq_overlap(test_df)

        # Print the class samples after overlap removal
        # print("test dataset class distribution:")
        # for label in labels_list:
        #     print(f"{label}: ", test_df[label].value_counts().to_dict())

        # Apply aforementioned data preprocessing functions
        test_df["text"] = test_df["text"].apply(
            lambda x: self.remove_contractions(self.remove_urls(self.remove_twitter_handles(x)))
        )

        return test_df

    def compute_weights(self):
        """
        This function calculates the weights for the training dataset
        in order to apply oversampling to minority classes when instantiating
        the dataset in the setup method using Torch's WeightedRandomSampler class.
        """

        # Identify list of labels
        labels_list = [
            "sarcasm",
            "irony",
            "satire",
            "understatement",
            "overstatement",
            "rhetorical_question",
        ]
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
            self.train_data, self.val_data = self.load_and_preprocess_training_data()
            # Augment the training dataset
            self.train_data = self.augment_data(self.train_data)
            # create a dict of the training data
            self.data_train = self.train_data.to_dict("records")
            # create a dict of the val data
            self.data_val = self.val_data.to_dict("records")
            # set label list for identifying class sample distribution
            labels_list = [
                "sarcasm",
                "irony",
                "satire",
                "understatement",
                "overstatement",
                "rhetorical_question",
            ]
            # print the class distributions after augmentation/overlap etc has
            # been applied
            # print("Train data class distributions:")
            # for label in labels_list:
            #     print(f"{label}: ", self.train_data[label].value_counts().to_dict())
            # print("\nValidation data class distributions:")
            # for label in labels_list:
            #     print(f"{label}: ", self.val_data[label].value_counts().to_dict())

            # compute the class weights
            self.compute_weights()

            # create training and validation datasets using the SubcategoryDataset
            # class
            self.train_dataset = SubcategoryDataset(self.data_train, self.tokenizer)
            self.val_dataset = SubcategoryDataset(self.data_val, self.tokenizer)

        if stage == "test":
            # If the stage is test, we load and preprocess test data
            self.test_data = self.load_and_preprocess_test_data()
            # create a dict of the test data
            self.data_test = self.test_data.to_dict("records")
            # create the test dataset using the SubcategoryDataset class
            self.test_dataset = SubcategoryDataset(self.data_test, self.tokenizer)

        if stage == "predict":
            # This is currently not used due to a lack of other data for inference
            # however the steps should be mostly the same as if stage == test.
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
    """

    def __init__(self, data_module=None, batch_size=None, num_labels=None, electra_classifier=None, learning_rate=None):
        """
        The constructor method for the class, called upon initialisation of the CEC class.
        This method sets up essential variables, including the model, the batch size,
        the classifier, and the learning rate. Super is called to initialise these
        variables first, as they are inherited from the ELECTRA Classifier class.
        """
        super().__init__(data_module=data_module, batch_size=batch_size, num_labels=num_labels)

        # This ensures that a model has actually been loaded from local files
        # based on the fine-tuned ELECTRA Classifier, returning an error
        # if the model classifier has not correctly been instantiated.
        if electra_classifier is not None:
            self.model = electra_classifier.model
            self.update_classification_head(num_labels)
        else:
            raise ValueError("An ElectraClassifier must be instantiated")

        # Identify hyperparameters to be saved (PTL doesn't provide functionality)
        # for all hparams to be saved.
        serialiazable_hparams = {"num_labels": num_labels, "batch_size": batch_size, "learning_rate": learning_rate}

        # save hparams
        self.save_hyperparameters(serialiazable_hparams)

        # This manually calculates the class weights of samples in the dataset,
        # to be used in the weighted loss function.

        # set up computation device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # manual input of class counts AFTER augmentation
        class_counts = np.array([2200, 2400, 1000, 400, 1450, 3900])
        # calculates inverse of class ocunts to give more weights to under-represented classes
        inverse_counts = 1 / class_counts
        # normalise inverse counts so that weights == 1
        weights = inverse_counts / np.sum(inverse_counts)
        # convert weights into tensor, mve to device
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)

        """
        Assignment of essential variables for the class: data module, model layers,
        classifier, warmup steps (not used but needed as the class is inherited),
        learning rate, lists for predictions/scores/metrics used later, number of 
        classes (num_labels), dropout, class weights, and loss.
        """
        self.data_module = data_module
        self.num_layers = len(list(self.parameters()))
        self.electra = electra_classifier
        self.warmup_steps = None
        self.learning_rate = learning_rate
        self.classifier = self.model.classifier
        self.predictions = []
        self.f1_macro_scores = []
        self.f1_classes_scores = []
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.15)
        self.weights = class_weights
        self.loss_fct = nn.CrossEntropyLoss(weight=self.weights)

        # Sets the base model layers to not trainable
        for param in self.electra.base_model.parameters():
            param.requires_grad = False

        # Sets the classifier layer to trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # Instantiating various metrics used during training, validation, and testing
        # Precision, recall, F1 for classes, and macro F1.
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
        # Return the model from the base ELECTRA Classifier so that all
        # model weights are the same following fine-tuning
        return self.model.electra

    def update_classification_head(self, num_labels):
        # Function to update the model's classification head to match the new
        # number of labels (6) as opposed to using a binary classification head
        # as in the inherited ELECTRA Classifier module.
        config = self.model.config
        config.num_labels = num_labels
        new_classifier = ElectraClassificationHead(config)
        self.model.classifier = new_classifier

    def train_dataloader(self):
        # get the train dataloader from the dataloader class
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        # get the validation dataloader from the dataloader class
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        # get the test dataloader from the dataloader class
        return self.data_module.test_dataloader()

    def predict_dataloader(self):
        # get the predict dataloader from the dataloader class
        return self.data_module.predict_dataloader()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        The forward pass method, which is automatically called by PTL during each
        step during model training, validation, and testing.
        """

        # Apply ELECTRA model to input data, taking outputs derived from input
        # tokens and attention mask.
        outputs = self.electra(input_ids, attention_mask=attention_mask)
        # Apply a dropout layer to output of ELECTRA model, setting some model weights
        # == 0
        dropout_outputs = self.dropout(outputs[0])
        # take the logits from classifier after applying the dropoout to
        # generate predictions.
        logits = self.classifier(dropout_outputs)

        # verify  that labels are provided
        if labels is not None:
            # calculate the loss between preds and true values
            loss = self.loss_fct(logits, labels.float())
            # gets index of max probability preds, transforming one-hot encoded
            # labels back to class indices
            loss_labels = torch.argmax(labels, dim=-1)
            # return the loss and logits for use in other methods
            return loss, logits
        else:
            return logits

    def on_train_start(self):
        # Verify the length of the training dataset
        self.train_dataset_len = len(self.train_dataloader().dataset)

    def on_train_batch_start(self, batch, batch_idx):
        # override method from ELECTRA Classifier class
        pass

    def on_validation_epoch_end(self):
        # override method from ELECTRA Classifier class
        pass

    def unfreeze_next_layer(self):
        # override method from ELECTRA Classifier class
        pass

    def training_step(self, batch, batch_idx):
        """
        Training step is required for PTL functionality. It
        is the logic for a single iteration during model training. Called every
        batch.
        """
        # Extract inputs from the batch to be fed to the model
        input_ids, attention_mask, labels = batch
        # squeeze labels to remove extra dimensions and ensure they match expected
        labels = labels.squeeze(1)
        # Call forward pass method to get loss & logits
        loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Take maximum value among predictions for preds (ensuring there is only 1)
        preds = torch.argmax(logits, dim=-1)
        # Take maximum value among labels for labels (ensuring there is only 1)
        labels = torch.argmax(labels, dim=-1)

        # Logging - here we log the F1 macro & for classes, with overall precision
        # & recall to be used in assessing the model during training.
        # Logged at each step & epoch.
        f1_macro = self.train_f1_macro(preds, labels)
        f1_classes = self.train_f1_classes(preds, labels)
        prec = self.train_precision(preds, labels)
        rec = self.train_recall(preds, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_precision", prec, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_recall", rec, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1_macro", f1_macro, on_step=True, on_epoch=True, prog_bar=True)

        # Logging the F1 score for each class
        for i, score in enumerate(f1_classes):
            self.log(f"Class {i} F1", score, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step is required for PTL functionality. It
        is the logic for a single iteration during the model validation step.
        Called every batch.
        """
        # Extract inputs from the batch to be fed to the model
        input_ids, attention_mask, labels = batch
        # squeeze labels to remove extra dimensions and ensure they match expected
        labels = labels.squeeze(1)
        # Call forward pass method to get loss & logits
        loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Take maximum value among predictions for preds (ensuring there is only 1)
        preds = torch.argmax(logits, dim=-1)
        # Take maximum value among labels for labels (ensuring there is only 1)
        labels = torch.argmax(labels, dim=-1)

        # Logging - here we log the F1 macro & for classes, with overall precision
        # & recall to be used in assessing the model during validation.
        # Logged at each step & epoch.
        f1_macro = self.val_f1_macro(preds, labels)
        f1_classes = self.val_f1_classes(preds, labels)
        prec = self.val_precision(preds, labels).mean()
        rec = self.val_recall(preds, labels).mean()
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        self.log("val_precision", prec, on_step=True, on_epoch=True)
        self.log("val_recall", rec, on_step=True, on_epoch=True)
        self.log("val_f1_macro", f1_macro, on_step=True, on_epoch=True)

        # Logging the F1 score for each class
        for i, score in enumerate(f1_classes):
            self.log(f"Class {i} F1", score, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Test step is required for PTL functionality. It
        is the logic for a single iteration during the model test step.
        Called every batch (only 1 batch in test dataset).
        """
        # Extract inputs from the batch to be fed to the model
        input_ids, attention_mask, labels = batch
        # squeeze labels to remove extra dimensions and ensure they match expected
        labels = labels.squeeze(1)
        # Call forward pass method to get loss & logits
        loss, logits = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Take maximum value among predictions for preds (ensuring there is only 1)
        preds = torch.argmax(logits, dim=-1)
        # Take maximum value among labels for labels (ensuring there is only 1)
        labels = torch.argmax(labels, dim=-1)

        # Logging - here we log the F1 macro & for classes, to be used in
        # assessing the model during testing. Logged at each step.

        f1_macro = self.test_f1_macro(preds, labels)
        f1_classes = self.test_f1_classes(preds, labels)

        # Append values to the class lists self.predictions, targets, f1_macro_scores,
        # f1_classes_scores to be used during the test function later.
        self.predictions.append(preds.detach().cpu())
        self.targets.append(labels.detach().cpu())
        self.f1_macro_scores.append(f1_macro.detach().cpu())
        self.f1_classes_scores.append(f1_classes.detach().cpu())

        self.log("test_loss", loss, on_step=True, prog_bar=True)
        self.log("test_f1_macro", f1_macro)

        # Logging the F1 score for each class
        for i, score in enumerate(f1_classes):
            self.log(f"Class {i} F1", score, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_start(self):
        # This function ensures that the predictions & targets lists are
        # empty at the start of each test step
        self.predictions = []
        self.targets = []

    def on_test_epoch_end(self):
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
        """
        This function isn't actually used, but is useful if we want to test
        the model on some unseen data. It is the same as the test function.
        """
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
        # Similar function for ensuring that the predictions & targets lists
        # are empty at the start of the predict step.
        self.predictions = []
        self.targets = []

    def collate_fn(self, batch):
        # Function to ensure batching is done correctly
        inputs, labels = zip(*batch)
        return torch.stack(inputs), torch.stack(labels)

    @property
    def current_lr(self):
        # Define the current learning rate
        return self.optimizers().param_groups[0]["lr"]

    def configure_optimizers(self):
        """
        This function implements the optimizer for training. It currently uses RAdam,
        implementation also experimented with using variable learning rates
        and learning rate schedulers, but for best performance RAdam with
        weight decay seems to work best.
        """

        optimizer = RAdam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)
        return optimizer

    def __getstate__(self):
        # Function for getting the state of the model.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Function for setting the state of the model.
        self.__dict__.update(state)


class MetricsCallback(Callback):
    """
    Custom MetricsCallback class - a subclass of PTL's 'callback' class. This
    records various metrics at different points during training and validation
    steps.
    """

    def __init__(self):
        # Constructor method for instantiating the lists of metrics
        super().__init__()
        self.train_metrics = []
        self.val_metrics = []
        self.batch_train_metrics = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # This function appends the trainer's callback_metrics at the end of
        # each training batch.
        self.batch_train_metrics.append(trainer.callback_metrics)

    def on_train_epoch_end(self, trainer, pl_module):
        # This function appends the trainer's callback_metrics at the end of
        # each training epoch.
        # Calculates the average of each metric  over all batches in the epoch
        # stores these metrics in epoch_train_metrics
        epoch_train_metrics = {}
        for key in self.batch_train_metrics[0].keys():
            epoch_train_metrics[key] = torch.stack([x[key] for x in self.batch_train_metrics]).mean()
        self.train_metrics.append(epoch_train_metrics)
        # reset the batch_train_metrics at the end of each training epoch
        self.batch_train_metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        # This function appends the trainer's callback_metrics at the end of
        # each validation epoch.
        self.val_metrics.append(trainer.callback_metrics)


def find_latest_checkpoint(version_prefix="sarcasm_detection_finetune_ckpt_v"):
    # This function returns the latest model checkpoint saved locally,
    # by checking the checkpoint directory for the file with the correct
    # version prefix and file type.
    checkpoints = [
        file for file in os.listdir(checkpoint_directory) if file.startswith(version_prefix) and file.endswith(".ckpt")
    ]
    # Return the latest version of fine-tuned ELECTRA
    return (
        max(checkpoints, key=lambda x: os.path.getctime(os.path.join(checkpoint_directory, x))) if checkpoints else None
    )


def load_model(transfer_data_module, saved_data_module):  # also saved_data_module
    # This function loads in the latest model version found by find_latest_checkpoint
    # and instantiates a version of this model for our task (multi-class classification)
    # it also uses the correct data module, delineated as the transfer_data_module
    latest_checkpoint = find_latest_checkpoint()

    if latest_checkpoint:
        full_checkpoint_path = os.path.join(checkpoint_directory, latest_checkpoint)

        # load model
        loaded_model = ElectraClassifier.load_from_checkpoint(full_checkpoint_path, data_module=saved_data_module)

        # new model with different classification head
        transfer_model = CustomElectraClassifier(
            electra_classifier=loaded_model,
            data_module=transfer_data_module,
            num_labels=6,
            batch_size=transfer_data_module.batch_size,
            learning_rate=5e-6,
        )

        # load the model's state dict from the fine-tuned ELECTRA Classifier.
        transfer_model.model.electra.load_state_dict(loaded_model.model.electra.state_dict())

        # return the model.
        return transfer_model

    else:
        print("No model checkpoint found")


def fit(model, data_module):
    """
    Fit is a required function for PTL. This instantiates the trainer instance,
    and allows implementation of the learning rate monitor, the TensorBoard logger,
    early stopping.
    """

    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=logdir, name=f"esubcat_model_v{sub_version_number}")
    metrics_callback = MetricsCallback()
    early_stopping = EarlyStopping("val_loss", patience=5, verbose=True, mode="min")

    # Instantiate the trainer
    trainer = Trainer(
        max_epochs=100,
        callbacks=[
            lr_monitor,
            metrics_callback,
            early_stopping,
        ],
        logger=logger,
    )

    # Call 'fit', or train

    trainer.fit(model, data_module)

    # save a model checkpoint

    trainer.save_checkpoint(sub_checkpoint_path)
    return model


def test(model, data_module):
    # This outlines the test function for the model, and is required by
    # PTL if you want to test.
    trainer = Trainer()

    # testing
    test_result = trainer.test(model, data_module)
    # return the lists of predictions and class/macro f1 scores
    return model.predictions, model.f1_macro_scores, model.f1_classes_scores


def get_f1_scores(predictions, macro_f1_scores, classes_f1_scores):
    # function for obtaining the model's macro and class f1 scores
    macro_f1_scores = torch.mean(torch.stack(macro_f1_scores))
    classes_f1_scores = torch.mean(torch.stack(classes_f1_scores))
    return macro_f1_scores, classes_f1_scores


def predict(model, data_module):
    # function for predicting with this model. Instantiates trainer.
    # Currently not used.
    trainer = Trainer()

    predict_result = trainer.predict(model, data_module)

    print(predict_result)

    f1_scores = get_f1_scores(predict_result)
    print(f"f1 score for predictions: {torch.IntTensor.item(f1_scores)}")


def launch_tensorboard(logdir):
    # function for launching tensorboard for model performance visualisation.
    command = f"tensorboard --logdir={logdir}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return process


def tensor_to_native(obj):
    # Function to convert PyTorch tensors in an object into Python data structures
    # Used in main method to be able to get model predictions after testing.
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_native(v) for v in obj]
    else:
        return obj


def main():
    # Main method
    # Instantiate data modules - including ELECTRA Classifier (saved) and the
    # required data module for this multi-class model (sub)
    saved_data_module = SarcasmDataModule(data_path=data_path, batch_size=16)
    sub_data_module = SarcasmSubDataModule(data_path=[sub_data_path_train, sub_data_path_test], batch_size=16)
    # Create a new instance of the model using the load_model function
    transfer_model = load_model(saved_data_module=saved_data_module, transfer_data_module=sub_data_module)
    # wrapped the training/testing in a conditional, to ensure that the model
    # is being correctly loaded
    if transfer_model is not None:
        # log directory for output during training
        logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"
        # launch tensorboard
        tensorboard_process = launch_tensorboard(logdir)
        # fit the model
        transfer_model = fit(transfer_model, sub_data_module)
        # end tensorboard logging
        tensorboard_process.terminate()
        # get the predictions and f1 scores through testing the model
        predictions, macro_f1_scores, classes_f1_scores = test(transfer_model, sub_data_module)
        # convert tensor predictions to readable format
        predictions_list = tensor_to_native(predictions)
        # get the f1 scores
        f1_scores = get_f1_scores(predictions, macro_f1_scores, classes_f1_scores)
        # log the predictions in a new .json
        with open("predictions.json", "w") as f:
            json.dump(predictions_list, f)

        print(f1_scores)

    else:
        print("failed to load the transfer model")


# call main method
if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
