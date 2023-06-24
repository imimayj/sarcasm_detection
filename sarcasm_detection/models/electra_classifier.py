"""
This is the ELECTRA Classifier module, which is fine-tuned on the binary task of 
sarcasm detection. This module utilises the News Headlines Dataset for Sarcasm Detection,
and the top 2% of the base model's layers are unfrozen gradually.
"""

import datetime
import os
import subprocess

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, Precision, Recall
from torchmetrics.classification import BinaryF1Score
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from transformers.models.electra.modeling_electra import ElectraClassificationHead

# Defining global variables

data_path = "/workspaces/sarcasm_detection/sarcasm_detection/project_data/Sarcasm_Headlines_Dataset_v2.json"
sub_data_path_train = "/workspaces/sarcasm_detection/notebooks/project_data/train.csv"
sub_data_path_test = "/workspaces/sarcasm_detection/notebooks/project_data/test.csv"
version_number = 8
sub_version_number = 9
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = f"/workspaces/sarcasm_detection/sarcasm_detection/checkpoints/fine_tuned/sarcasm_detection_finetune_ckpt_v{version_number}_{current_time}.ckpt"
sub_checkpoint_path = f"/workspaces/sarcasm_detection/notebooks/checkpoints/custom_trained/subcat_finetune_ckpt_v{sub_version_number}_{current_time}.ckpt"
checkpoint_directory = os.path.dirname(checkpoint_path)
logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"
save_directory = (
    f"/workspaces/sarcasm_detection/sarcasm_detection/saved_models/sarcasm_model_v{version_number}_{current_time}"
)


class SarcasmDataset(Dataset):
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
        # ELECTRA Classifier module
        text = self.data[idx]["headline"]
        labels = self.data[idx]["is_sarcastic"]
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
            encodings["input_ids"].flatten(),
            encodings["attention_mask"].flatten(),
            torch.tensor(labels),
        )


class SarcasmDataModule(pl.LightningDataModule):
    """
    This is the Datamodule class, which is used to prepare and preprocess the
    Semeval dataset. The data is split into training, validation, testing, and
    prediction datasets depending on the stage passed to the Datamodule class
    by PyTorch Lightning's Trainer class.
    """

    def __init__(self, data_path, batch_size, tokenizer="google/electra-small-discriminator"):
        """
        The constructor method for the class, called upon initialisation of a new
        class. This method sets up essential variables, including the data path,
        pretrained ELECTRA tokenizer, batch size, and collate function for data
        batching
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = ElectraTokenizer.from_pretrained(tokenizer)
        self.batch_size = batch_size
        self.collate_fn = self.default_collate_fn

    def prepare_data(self):
        """
        This function prepares the data for input to the model. It sets the column
        types, dropping unnecessary columns, and splitting the dataset into
        training/validation/test datasets using the split_datasets function below.
        The dataset is designed for this application, so there are very few
        data integrity issues
        """
        col_types = {"headline": "str", "is_sarcastic": "int32"}

        df = pd.read_json(self.data_path, lines=True).drop(columns=["article_link"]).astype(col_types)

        train_df, val_df, test_df = self.split_datasets(df)

        # return the datasets as dicts
        self.data_train = train_df.to_dict("records")
        self.data_val = val_df.to_dict("records")
        self.data_test = test_df.to_dict("records")

    def setup(self, stage: str = None):
        """
        Setup method for establishing which datasets to load using the SarcasmDataset
        module based on the stage passed to the datamodule by PyTorch Lightning's
        Trainer class.
        """
        if stage == "fit":
            # If the trainer stage is fit, we create the training/validation
            # datasets
            self.train_dataset = SarcasmDataset(self.data_train, self.tokenizer)
            self.val_dataset = SarcasmDataset(self.data_val, self.tokenizer)

        if stage == "test":
            # If the trainer stage is test, we create the test dataset using
            # the SubcategoryDataset class
            self.test_dataset = SarcasmDataset(self.data_test, self.tokenizer)

        if stage == "predict":
            # If the trainer stage is predict, we create the predict dataset using
            # the SubcategoryDataset class
            self.predict_dataset = SarcasmDataset(self.data_test, self.tokenizer)

    def steps_per_epoch(self):
        # define the steps_per_epoch based on length of training dataset
        return len(self.train_dataset)

    def split_datasets(self, df):
        # Splits the dataset into 80/20 training/temporary dataset
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        # splitting the temp dataset into 10/10 validation and testing datasets
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)
        return train_df, val_df, test_df

    def train_dataloader(self):
        # This instantiates the dataloader for training dataset, which is required
        # for correct PyTorch lightning model implementation to load the correct
        # dataset.
        # The train dataset applies the weights identified above to create a
        # weighted random sampler instance to oversample minority classes.
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=6, shuffle=True)

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
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            shuffle=False,
        )

    def default_collate_fn(self, batch):
        # This function creates the batch by combining multiple data samples
        # comprising the text, attention mask, and labels.
        input_ids, attention_mask, labels = zip(*batch)
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)
        return input_ids, attention_mask, labels


class ElectraClassifier(pl.LightningModule):
    """
    This is the ELECTRA Classifier class, used for outlining
    how the model used will be implemented, providing information to the Trainer
    so that training/validation/testing is executed in the correct way.
    """

    def __init__(
        self,
        data_module,
        batch_size,
        model_name="google/electra-small-discriminator",
        num_labels=None,
        learning_rate=2e-5,
    ):
        """
        The constructor method for the class, called upon initialisation of the class.
        This method sets up essential variables, including the model name, the batch size,
        the classifier, the number of labels and the learning rate.
        Super is called to initialise these variables first.
        """
        super().__init__()

        # Identify hyperparameters to be saved (PTL doesn't provide functionality)
        # for all hparams to be saved. Save the hparams.
        serialiazable_hparams = {"num_labels": num_labels, "batch_size": batch_size, "learning_rate": learning_rate}
        self.save_hyperparameters(serialiazable_hparams)
        """
        Assignment of essential variables for the class: model name, model,
        data module, number of layers, warmup steps, learning rate, batch size,
        number of labels for classification, validation metrics for performance
        monitoring, patience, and the unfreeze step/current unfreeze index, used
        later in unfreezing the base model layers when fine tuning. 
        """
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
        self.current_unfreeze_idx = int(len(list(self.model.electra.parameters())))
        self.unfreeze_step = 1

        # set the base model layers to not trainable
        for param in self.model.electra.parameters():
            param.requires_grad = False

        # Instantiating various metrics used during training, validation, and testing
        # such as accuracy, precision, recall, and f1 score.
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
        # Forward pass, returning the output of the model following input of data
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def on_train_start(self):
        # Verify the length of the training dataset
        self.train_dataset_len = len(self.train_dataloader().dataset)

    def on_train_batch_start(self, batch, batch_idx):
        """
        This method unfreezes certain base ELECTRA layers during training given
        that the training has met certain conditions.
        """
        # If the model has reached the global warmup steps of 10,000
        if self.global_step == self.warmup_steps:
            # identify the current index of layers unfrozen in ELECTRA base model
            self.current_unfreeze_idx = int(len(list(self.model.electra.parameters())) * 0.9)
            # call the unfreeze next layer function
            self.unfreeze_next_layer()

            # get the optimizer for certain layers
            optimizer = self.optimizers()
            freeze_idx = int(len(list(self.model.electra.parameters())) * 0.9)

            # This part of the function ensures that the correct amount of layers
            # i.e. less than 2% of total layers working from the top down
            # gets unfrozen if the threshold of warmup steps has been passed.
            # it also sets the learning rate of the frozen layers to 0
            for idx, param_group in enumerate(optimizer.param_groups):
                if self.global_step < self.warmup_steps:
                    if idx < freeze_idx:
                        param_group["lr"] = 0

    def training_step(self, batch, batch_idx):
        """
        Training step is required for PTL functionality. It
        is the logic for a single iteration during model training. Called every
        batch.
        """
        # Extract inputs from the batch to be fed to the model
        input_ids, attention_mask, labels = batch
        # Call forward pass method to get loss, logits, and preds
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)

        # Logging - here we log the accuracy, precision, recall, and training loss
        acc = self.train_accuracy(preds, labels)
        prec = self.train_precision(preds, labels)
        rec = self.train_recall(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_precision", prec, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_recall", rec, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step is required for PTL functionality. It
        is the logic for a single iteration during the model validation step.
        Called every batch.
        """
        # Extract inputs from the batch to be fed to the model
        input_ids, attention_mask, labels = batch
        # Call forward pass method to get loss, logits, and preds
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        # Logging - here we log the accuracy, precision, recall, and validation loss
        acc = self.val_accuracy(preds, labels)
        prec = self.val_precision(preds, labels)
        rec = self.val_recall(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", prec, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recall", rec, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss = loss.item()

    def on_validation_epoch_end(self):
        """
        This function monitors the model's base performance at the end of each
        validation epoch. This means that the base model's layers are only
        unfrozen when the performance of validation loss has not improved within
        the period of patience - set at 2 epochs.
        """
        current_val_performance = self.val_loss
        if current_val_performance > self.best_val_performance:
            self.best_val_performance = current_val_performance
            self.epochs_since_best_performance = 0
        else:
            self.epochs_since_best_performance += 1

        if self.epochs_since_best_performance >= self.patience:
            self.unfreeze_next_layer()

    def unfreeze_next_layer(self):
        """
        This function unfreezes the next layer in the ELECTRA base model.
        """
        # calculate the number of total layers in the base model
        num_params = len(list(self.model.electra.parameters()))
        # find the total amount of layers we want to unfreeze
        freeze_idx = int(num_params * 0.9)
        # find the current layer which has or has not been unfrozen
        unfreeze_idx = self.current_unfreeze_idx

        # this unfreezes the next layer in the base model only if the current
        # layer index is within the parameters of the top 2% of the model's layers.
        # it then increases the current unfreeze index.
        if unfreeze_idx < freeze_idx:
            self.current_unfreeze_idx += self.unfreeze_step

            for idx, param in enumerate(self.model.electra.parameters()):
                if idx >= unfreeze_idx and idx < self.current_unfreeze_idx:
                    param_requires_grad = True
            # print(f"unfroze layers up to idx {self.current_unfreeze_idx}")

    def test_step(self, batch, batch_idx):
        """
        Test step is required for PTL functionality. It
        is the logic for a single iteration during the model test step.
        Called every batch (only 1 batch in test dataset).
        """
        # Extract inputs from the batch to be fed to the model
        input_ids, attention_mask, labels = batch
        # Call forward pass method to get loss, logits & preds
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = outputs.logits.argmax(dim=-1)
        # Logging - here we log the accuracy and f1 score, and append predictions
        # and target labels to our lists
        acc = self.test_accuracy(preds, labels)
        self.predictions.append(preds.detach().cpu())
        self.targets.append(labels.detach().cpu())
        f1_score = self.f1(preds, labels)
        # print("logits test_step:", outputs.logits)
        # print("predictions test_step:", preds)
        self.log("test_f1", f1_score, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_start(self):
        # This function ensures that the predictions & targets lists are
        # empty at the start of each test step
        self.predictions = []
        self.targets = []

    def on_test_epoch_end(self):
        # This function provides functionality to extract the predictions and
        # labels, which is useful for assessing model prediction performance
        # print("predictions on_test_epoch_end:", self.predictions)
        # print("targets on_test_epoch_end", self.targets)
        all_preds = torch.cat(self.predictions)
        all_labels = torch.cat(self.targets)
        return {"preds": all_preds, "labels": all_labels}

    def predict_step(self, batch, batch_idx):
        """
        This function isn't actually used, but is useful if we want to test
        the model on some unseen data. It is the same as the test function.
        """
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = outputs.logits.argmax(dim=-1)
        self.predictions.append(preds.detach().cpu())
        self.targets.append(labels.detach().cpu())
        f1_score = self.f1(preds, labels)
        # print("logits test_step:", outputs.logits)
        # print("predictions test_step:", preds)
        return preds, f1_score

    def on_predict_start(self):
        # Same as on_test_start
        self.predictions = []
        self.targets = []

    def collate_fn(self, batch):
        # Function to ensure batching is done correctly
        inputs, labels = zip(*batch)
        return torch.stack(inputs), torch.stack(labels)

    @property
    def config(self):
        # function for returning the ELECTRA model config - used when saving
        # and loading models
        return self.model.config

    @property
    def current_lr(self):
        # return the current learning rate
        return self.optimizers().param_groups[0]["lr"]

    def configure_optimizers(self):
        """
        This function configures the optimizers for model training.
        It uses a variable learning rate (CosineAnnealingWarmRestarts) as part
        of a scheduled learning rate which changes following the initial 10,000
        warmup steps
        """
        # instantiate RAdam optimizer
        optimizer = RAdam(self.parameters(), lr=self.hparams.learning_rate)

        # Calculate total number of training steps
        num_training_steps = len(self.train_dataloader()) * self.trainer.max_epochs

        # assign variables required for CosineAnnealingWarmRestarts
        T_0 = num_training_steps - self.warmup_steps
        T_mult = 1
        eta_min = 0

        # instantiate the learning rate scheduler
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

        # define the warmup schedule for changing the learning rate
        scheduler_with_warmup = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "warmup_steps": self.warmup_steps,
            "warmup_start_lr": 0,
        }

        # return the optimizer, scheduler, and validation loss monitor
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_with_warmup,
            "monitor": "train_loss",
        }

    def update_classification_head(self, num_labels):
        # function for updating the classification head of this model
        # used when re-loading this model after training into the Custom ELECTRA
        # Classifier and Aggregate-ELECTRA.
        self.model.classifier = ElectraClassificationHead(self.model.config, num_labels)
        self.num_labels = num_labels

    def __getstate__(self):
        # Function for getting the state of the model. Prevents conflicts
        # when reloading this model
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Function for setting the state of the model. Prevents conflicts
        # when reloading this model
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
        self.batch_train_metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        # This function appends the trainer's callback_metrics at the end of
        # each validation epoch.
        self.val_metrics.append(trainer.callback_metrics)


def fit(model, data_module):
    """
    Fit is a required function for PTL. This instantiates the trainer instance,
    and allows implementation of the learning rate monitor, the TensorBoard logger,
    and early stopping.
    """
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=logdir, name=f"electra_model_v{version_number}")
    metrics_callback = MetricsCallback()
    early_stopping = EarlyStopping("val_loss", patience=5, verbose=True)

    # Instantiate the trainer
    trainer = Trainer(
        max_epochs=10000,
        callbacks=[
            lr_monitor,
            metrics_callback,
            early_stopping,
        ],
        logger=logger,
    )

    # Call 'fit', or train
    trainer.fit(model, data_module)

    # save the model checkpoint for later use
    trainer.save_checkpoint(checkpoint_path)
    # ensure that the base model (without classifier) can be reloaded when used
    # by other classes
    base_model = model.model.base_model
    # save the pretrained model to the correct directory
    base_model.save_pretrained(save_directory)
    return model


def test(model, data_module):
    # This outlines the test function for the model, and is required by
    # PTL if you want to test.
    trainer = Trainer()
    test_result = trainer.test(model, data_module)
    # here we print the test result for debugging and assessing the model
    # performance
    print(test_result)


def get_f1_scores(predict_result):
    # get the f1 scores from results.
    f1_scores = [f1_score for (_batch_preds, f1_score) in predict_result]
    f1_scores = torch.mean(torch.stack(f1_scores))
    return f1_scores


def predict(model, data_module):
    # function for predicting with this model. Instantiates trainer.
    # Currently not used.
    trainer = Trainer()

    predict_result = trainer.predict(model, data_module)

    # print(predict_result)

    f1_scores = get_f1_scores(predict_result)
    # print(f"f1 score for predictions: {torch.IntTensor.item(f1_scores)}")


def launch_tensorboard(logdir):
    # function for launching tensorboard for model performance visualisation.
    command = f"tensorboard --logdir={logdir}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return process


def main():
    # Main method
    # Instantiate data module - SarcasmDataModule, and the model - ElectraClassifier.
    # trains the model, tests the model.
    logdir = "/workspaces/sarcasm_detection/sarcasm_detection/tb_logs"
    tensorboard_process = launch_tensorboard(logdir)
    data_module = SarcasmDataModule(data_path=data_path, batch_size=16)
    model = ElectraClassifier(data_module=data_module, batch_size=data_module.batch_size, num_labels=2)
    model = fit(model, data_module)

    tensorboard_process.terminate()

    model = test(model, data_module)


if __name__ == "__main__":
    main()
