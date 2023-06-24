## The ELECTRA Model for Sarcasm Subcategory Classification
## Dissertation Project June 2023

## Description

This repository contains the files produced for assessing the efficacy of the ELECTRA pre-trained large language model on the task of multi-class sarcasm subcategory classification. The repository contains the python files required for running the various models implemented during the course of the project. This includes five modules used during implementation, namely: Aggregate_electra.py, custom_electra_classifier.py, electra_classifier.py, 7_class_electra.py, and electra_no_FT.py. The first model - electra_classifier.py - is the model which is fine-tuned to generalise on sarcastic data. The second model - custom_electra_classifier.py - is the module which re-loads the saved model from electra_classifier.py on the Semeval 2022 dataset for sarcasm subcategory classification. The third model - aggregate_electra.py - is the module which also re-loads the saved model from electra_classifier.py on the Semeval 2022 datasets, but aggregates the minority classes (irony, satire, understatement, overstatement, and rhetorical question) into one 'other' class, while creating a not_sarcastic class for the non-sarcastic samples included in the dataset. The fourth model - 7_class_electra.py - is an experimental implementation used for creating a not_sarcastic class used in addition with the other subcategories to assess its results in comparison with custom_electra_classifier.py. The final model - electra_no_FT.py - uses the base ELECTRA model loaded from HuggingFace to assess the base model's performance on the Semeval 2022 datasets in comparison to the custom_electra_classifier.py. 

## Installation 

This project is configured to be run in a Docker container. To aid with setup, a Visual Studio Code devcontainer has been created. To run the project, simply open it using the VSCode devcontainer extension. 

For a non-devcontainer approach, Python dependencies can be installed by running:

```
$ pip install -r .devcontainer/requirements.txt
```

### Note 

This project has been developed on a system using an NVIDIA GPU. If not running an NVIDIA GPU and the devcontainer approach is used, remove the CUDA installation feature in the .devcontainer/devcontainer.json file:

```
"features": {
		"ghcr.io/devcontainers/features/nvidia-cuda:1": {
			"installCudnn": true
		}
```

As well as removing the index url for PyTorch installation:

```
--index-url https://download.pytorch.org/whl/cu118
```

## Datasets

The datasets are not included in the file submission following discussions with my supervisor. As these datasets are readily available in open source, they can be downloaded from the following links:

For the News Headlines Dataset for Sarcasm Detection:

https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection 

For the Semeval datasets:

https://github.com/iabufarha/iSarcasmEval 

--> The datasets to be downloaded from the iSarcasmEval repo are iSarcasmEval -> train -> train.En.csv and iSarcasmEval -> test -> task_B_En_test.csv. 

## Note

The dataset used for this task was version 2 of the Sarcasm Headlines Dataset for Sarcasm Detection, which is included in the .zip file when downloaded from the Kaggle page. Both work for the purposes of this project, but version 2 includes slightly more examples and was therefore deemed more suitable.

All three datasets should be saved in /sarcasm_detection/sarcasm_detection/project_data in order for global variables assigned in the code to work correctly.

- The Semeval training dataset should be saved as 'sarcasm_detection/sarcasm_detection/project_data/train.csv'.
- The Semeval test dataset should be saved as 'sarcasm_detection/sarcasm_detection/project_data/test.csv'.
- The News Headlines Dataset for Sarcasm Detection should be saved as 'sarcasm_detection/sarcasm_detection/project_data/Sarcasm_Headlines_Dataset_v2.json'.

This is to ensure that the datamodules in all classes can correctly access the datasets at the correct time. These paths are assigned as global variables throughout the modules.

## Saved models

The modules aggregate_electra.py, custom_electra_classifier.py, and 7_class_electra.py rely on a saved version of the fine-tuned ELECTRA model, as trained in the electra_classifier.py module. The most recent saved version of this model can be accessed at this address: https://drive.google.com/drive/folders/199qmIT-G_c1AesTb2PnZ6XaCExRYzX3v?usp=sharing. This model has not been uploaded to Engage as it is very large.

The model, named 'sarcasm_detection_finetune_ckpt_v8_20230623-090427.ckpt' can be downloaded. It must then be added into this location:

/sarcasm_detection/sarcasm_detection/checkpoints/fine_tuned/{file} 

So that the models which require this loaded version of the model config can load them in for training and testing.

## Note

Please do not change the name of the above file, otherwise the modules will not be able to access the information in the file correctly.

## Run

To run the main models evaluated during this project, please follow these steps:

1. It is recommended to start by running the electra_classifier.py model, which uses the News Headlines Dataset for Sarcasm Detection, and fine-tunes the outermost layers of the base ELECTRA model for generalisation on sarcastic data. Running the .py file will train and test this model.

2. Run the custom_electra_classifier.py model, which loads in the saved version of electra_classifier.py to be trained and tested on the Semeval 2022 datasets.

3. Run the aggregate_electra.py model, which loads in the saved version of electra_classifier.py and trains and tests it on three categories created from the Semeval datasets: 'sarcasm', 'not_sarcastic', and 'other'.

## Experimental models for comparison:

These models are included as a reference point for comparison of results with those obtained by custom_electra_classifier.py. It is not entirely necessary to run these models, however they provide some other insights regarding ELECTRA's behaviour:

1. Run the 7_class_electra.py model, which is used for evaluation against the results of the custom_electra_classifier and includes a 'not_sarcastic' class on top of the 6 subcategories of 'sarcasm', 'irony', 'satire', 'understatement', 'overstatement', 'rhetorical_question'. 

2. Run the electra_no_FT.py model, which is used as a comparison between the performance of the base ELECTRA model with no fine tuning, and the custom_electra_classifier.py, which is a re-loaded version of the ELECTRA base model which has been fine-tuned to generalise on sarcastic data. 

## Note on memory

Many of the processes used in PyTorch Lightning automatically save logs and models to various folders within this repository. If your machine is low on memory and requires extra space when running, please delete the files generated when training and testing the above models from the following folders:

- /sarcasm_detection/lightning_logs
- /sarcasm_detection/sarcasm_detection/checkpoints/aggregate_trained
- /sarcasm_detection/sarcasm_detection/checkpoints/base
- /sarcasm_detection/sarcasm_detection/checkpoints/custom_trained
- /sarcasm_detection/sarcasm_detection/models/saved_models
- /sarcasm_detection/sarcasm_detection/tb_logs (This should only be deleted once results have been visualised through the Tensorboard port opened automatically when training commences).

Please do not clear the saved models from the /sarcasm_detection/sarcasm_detection/checkpoints/fine_tuned folder, as this will prevent several of the modules from running due to no model being loaded in. 

## Please contact me at ij321@bath.ac.uk should any issues surrounding running this project arise. 