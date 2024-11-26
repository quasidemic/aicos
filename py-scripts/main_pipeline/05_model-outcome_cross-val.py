#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
import pandas as pd
import json
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
import random
from random import sample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import torch
import logging
import numpy as np

project_dir = join('/work', 'aicos')
modules_p = join(project_dir, 'modules')
logs_dir = join(project_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

## LOGGING SETUP
logging.basicConfig(
    filename=join(logs_dir, 'crossval_binary_20241030.log'),  # Log file name
    filemode='w',        # Write mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.WARNING   # warning level - easiest way to avoid unnecessary prints
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
negatives_p = join(data_dir, 'negatives.csv')


## READ MAPPINGS
mapdf = pd.read_csv(mapdata_p) # mappings

## READ NEGATIVES
negdf = pd.read_csv(negatives_p)

## DATA HANDLING
cols_keep = ['Study ID', 'Verbatim Outcomes', 'Outcome Domains']
not_case_report_filter = mapdf['is_case_report'] == 0
mapdf_model = mapdf.loc[not_case_report_filter, cols_keep].rename(columns={"Verbatim Outcomes": "text", "Outcome Domains": "label"})
mapdf_model['label'] = "outcome"

### rename negatives df columns
negdf = negdf.rename(columns={"Verbatim Outcomes": "text", "Outcome Domains": "label"})
negdf['label'] = "not outcome"

## IDS FOR MODELLING
studyids = mapdf_model['Study ID'].unique().tolist()

## SEED USED FOR SAMPLING TRAINING, EVAL, TEST
seed_no = 4220

## ADD NEGATIVES
model_data = pd.concat([mapdf_model, negdf], axis=0).reset_index(drop = True)

## SET FIXED TEST SET
random.seed(seed_no)

test_size = 0.25
test_ids = sample(studyids, round(test_size * len(studyids)))

test_df = model_data.loc[model_data['Study ID'].isin(test_ids), ].drop(columns = ['Study ID'])
test_data = Dataset.from_pandas(test_df, preserve_index = False)

## REMAINING IDS
train_eval_ids = [id for id in studyids if id not in test_ids]

## TRAIN, EVAL DATA
n = 30 # using 30 articles based on 03_model-select-train

# split ids
ids_use = train_eval_ids[:n]

crossval_df = model_data.loc[model_data['Study ID'].isin(ids_use), :]

texts_crossval = crossval_df['text'].tolist()
labels_crossval = crossval_df['label'].tolist()


## MODEL, LABELS AND HYPERPARAMETERS
### model
model_name = "thenlper/gte-base"
### labels
labels = ['outcome', 'not outcome']
### device
device = "cuda" if torch.cuda.is_available() else "cpu"

### parameters from HPO
params = {
    'head_params': {
        'solver': 'liblinear',
        'max_iter': 100
    },
    'batch_size': 64,
    'num_epochs': 3,
    'body_learning_rate': 1.04e-05
    }

### Function to compute additional metrics
def compute_metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

## CROSS-VAL SETUP
logging.warning('Starting cross-validation...')

n_splits = 5 # n folds
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_no)

## Run cross-val
for fold, (train_index, val_index) in enumerate(kf.split(texts_crossval, labels_crossval), start = 1):
    print(f"Fold {fold}/{n_splits}")
    
    # Split fold into training and validation
    train_texts = [texts_crossval[i] for i in train_index]
    train_labels = [labels_crossval[i] for i in train_index]
    val_texts = [texts_crossval[i] for i in val_index]
    val_labels = [labels_crossval[i] for i in val_index]
    
    # Convert to Dataset
    fold_train_data = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    fold_val_data = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    
    # Load model with each fold
    model = SetFitModel.from_pretrained(
        model_name,
        head_params = params.get('head_params'),
        labels = labels
        ).to(device)
    

    # Setfit Training arguments
    args = TrainingArguments(
        batch_size = params.get('batch_size'),
        num_epochs = params.get('num_epochs'),
        body_learning_rate = params.get('body_learning_rate'),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        show_progress_bar=False
    )
    
    # Setfit Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=fold_train_data,
        eval_dataset=fold_val_data,
        metric="accuracy",
        column_mapping={"text": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
    )
    
    # Train model
    trainer.train()

    # Evaluate
    y_true = list(fold_val_data['label'])
    y_pred = model.predict(list(fold_val_data['text']))

    report = classification_report(y_true, y_pred, output_dict = True, zero_division = 0)

    # Write fold metrics to log
    logging.warning(f'Metrics for fold {fold}: {report}')

    # Write report to file
    out_path = join(output_dir, "cross-val_binary_model.json")

    try:
        with open(out_path, 'r') as f:
            # Load existing data as a list of dictionaries
            reports = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty list
        reports = []

    reports.append(report)

    with open(out_path, 'w') as f:
        json.dump(reports, f)

# Avg. metrics
avg_metrics = {metric: np.mean([m.get('macro avg')[metric] for m in reports]) for metric in ['precision', 'recall']}
logging.warning(f"Average metrics across {n_splits} folds: {avg_metrics}")