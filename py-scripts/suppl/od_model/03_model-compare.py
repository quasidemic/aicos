#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
import pandas as pd
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
import random
from random import sample
import torch
import logging
import json

project_dir = join('/work', 'aicos')
modules_p = join(project_dir, 'modules')
logs_dir = join(project_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

sys.path.append(modules_p)

## LOGGING SETUP
logging.basicConfig(
    filename=join(logs_dir, 'models-compare.log'),  # Log file name
    filemode='w',        # Write mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.WARNING   # Log level
)

logger = logging.getLogger()

## DIRS AND PATHS
data_dir = join(project_dir, 'data')
plot_dir = join(project_dir, 'plots')
output_dir = join(project_dir, 'output')
models_dir = join(project_dir, 'models')

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

mapdata_p = join(data_dir, 'articles_text_outcomes.csv')

## READ MAPPINGS
mapdf = pd.read_csv(mapdata_p) # mappings

## DATA HANDLING
cols_keep = ['Study ID', 'Verbatim Outcomes', 'Outcome Domains']
not_case_report_filter = mapdf['is_case_report'] == 0
mapdf_model = mapdf.loc[not_case_report_filter, cols_keep].rename(columns={"Verbatim Outcomes": "text", "Outcome Domains": "label"})

### fix labels
labels_fix = {
    'Adverse_x000D_ events': 'Adverse events',
    'Blood and lymphatic system _x000D_outcomes': 'Blood and lymphatic system outcomes',
    'Infection and infestation\n outcomes': 'Infection and infestation outcomes'   
}

mapdf_model['label'] = mapdf_model['label'].replace(labels_fix)
mapdf_model['label'] = mapdf_model['label'].str.strip()

### Labels to use
count_thres = 25
labels_filter = mapdf_model['label'].value_counts() >= count_thres
labels_keep = list(mapdf_model['label'].value_counts()[labels_filter].index)

### filter labels
row_labels_filter = mapdf_model['label'].isin(labels_keep)
mapdf_model = mapdf_model[row_labels_filter]

## IDS FOR MODELLING
studyids = mapdf_model['Study ID'].unique().tolist()

## SEED USED FOR SAMPLING TRAINING, EVAL, TEST
seed_no = 4220

## SET FIXED TEST SET
random.seed(seed_no)

test_size = 0.25
test_ids = sample(studyids, round(test_size * len(studyids)))

test_df = mapdf_model.loc[mapdf_model['Study ID'].isin(test_ids), ].drop(columns = ['Study ID'])
test_data = Dataset.from_pandas(test_df, preserve_index = False)

## REMAINING IDS
train_eval_ids = [id for id in studyids if id not in test_ids]

## FUNCTION FOR TRAINING, EVAL DATA BASED ON N ARTICLES
def set_train_eval(n, seed, eval_prop = 0.2, train_eval_ids = train_eval_ids, df_use = mapdf_model):

    # set seed
    random.seed(seed)
    
    # split ids
    ids_use = sample(train_eval_ids, n)

    train_prop = 1 - eval_prop
    train_ids = sample(ids_use, round(train_prop * len(ids_use)))
    eval_ids = list(set(ids_use) - set(train_ids))

    # set train, eval data based on ids
    train_df = df_use.loc[df_use['Study ID'].isin(train_ids), ].drop(columns = ['Study ID'])
    eval_df = df_use.loc[df_use['Study ID'].isin(eval_ids), ].drop(columns = ['Study ID'])
    
    # convert to transformers dataset
    train_data = Dataset.from_pandas(train_df, preserve_index = False)
    eval_data = Dataset.from_pandas(eval_df, preserve_index = False)

    return train_data, eval_data

## SET TRAIN AND EVAL DATA
train_data, eval_data = set_train_eval(50, seed_no)

## FUNCTION FOR METRICS
def compute_metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

## MODEL, LABELS AND HYPERPARAMETERS
### model
models = [
    "sentence-transformers/paraphrase-mpnet-base-v2",
    "NeuML/pubmedbert-base-embeddings",
    "thenlper/gte-base",
    "BAAI/bge-base-en-v1.5"
]
### labels
labels = labels_keep
### device
device = "cuda" if torch.cuda.is_available() else "cpu"

### parameters from HPO
params = {
    'head_params': {
        'solver': 'lbfgs',
        'max_iter': 200
    },
    'batch_size': 64,
    'num_epochs': 2,
    'body_learning_rate': 6.818e-05
    }

## FUNCTION FOR TRAINING MODEL
def train_wrapper(model_name, train_data = train_data, eval_data = eval_data, test_data = test_data, params = params):

    logger.warning(f"fitting model with {model_name}...")

    # Define the SetFit model
    model = SetFitModel.from_pretrained(
        model_name,
        head_params = params.get('head_params'),
        labels=labels
    ).to(device)

    # Define training arguments
    args = TrainingArguments(
        batch_size = params.get('batch_size'),
        num_epochs = params.get('num_epochs'),
        body_learning_rate = params.get('body_learning_rate'),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        metric=compute_metrics,
        column_mapping={"text": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    test_results = trainer.evaluate(test_data)

    # Store metrics
    metrics = {
        'model': model_name, 
        'accuracy': test_results['accuracy'],
        'precision': test_results['precision'], 
        'recall': test_results['recall'],
        'f1': test_results['f1']
    }

    return metrics


# Evaluate
eval_models = []

fileout = join(output_dir, 'models_eval_12-class.json')


for model_name in models:
    model_metrics = train_wrapper(model_name)

    # Add to JSON
    ## Open existing file
    try:
        with open(fileout, 'r') as f:
            data = json.load(f)  # Load the existing data into a Python object
    except FileNotFoundError:
        data = []  # If the file doesn't exist, start with an empty list

    ## Append the new metrics to the list
    if isinstance(data, list):
        data.append(model_metrics)

    ## Write the updated data back to the JSON file
    with open(fileout, 'w') as f:
        json.dump(data, f, indent=4)  # Save the updated data back to the file


    eval_models.append(model_metrics)

## store
fileout_csv = join(output_dir, 'models_eval_12-class.csv')
eval_models_df = pd.DataFrame.from_records(eval_models)
eval_models_df.to_csv(fileout_csv, index=False)