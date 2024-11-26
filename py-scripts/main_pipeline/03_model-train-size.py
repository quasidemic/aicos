#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
import pandas as pd
import re
import random
from random import sample, shuffle
import logging
import numpy as np
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
import json

project_dir = join('/work', 'aicos')
modules_p = join(project_dir, 'modules')
logs_dir = join(project_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

sys.path.append(modules_p)

from modelling import *

## LOGGING SETUP
logging.basicConfig(
    filename=join(logs_dir, 'eval_train_size_binary_20241021-1.log'),  # Log file name
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

## FUNCTION FOR TRAINING, EVAL DATA BASED ON N ARTICLES
def set_train_eval(n, seed, labels, eval_prop = 0.2, train_eval_ids = train_eval_ids, df_use = model_data):

    # set seed
    random.seed(seed)
    
    # split ids
    ids_use = train_eval_ids[:n]

    train_prop = 1 - eval_prop
    train_ids = sample(ids_use, round(train_prop * len(ids_use)))
    eval_ids = list(set(ids_use) - set(train_ids))

    # set train data based on ids
    train_df = df_use.loc[(df_use['Study ID'].isin(train_ids)), ]

    # set eval data based on ids and labels
    eval_df = df_use.loc[(df_use['Study ID'].isin(eval_ids)), ]
    
    # convert to transformers dataset
    train_data = Dataset.from_pandas(train_df, preserve_index = False)
    eval_data = Dataset.from_pandas(eval_df, preserve_index = False)

    return train_data, eval_data


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

## FUNCTION FOR MODELLING BASED ON N ARTICLES
def fit_model(n, seed, labels, model_name = model_name, params = params, device = device, test_data = test_data, logger = logger):

    logger.warning(f"Fitting model using {n} articles...")

    # Set train and eval data
    train_data, eval_data = set_train_eval(n, seed, labels)

    # Load SetFit model from Hub
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
        train_dataset=train_data,
        eval_dataset=eval_data,
        metric="accuracy",
        column_mapping={"text": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
    )

    # Train
    trainer.train()

    # Evaluate
    y_true = list(test_data['label'])
    y_pred = model.predict(list(test_data['text']))

    report = classification_report(y_true, y_pred, output_dict = True, zero_division = 0)
    cm = confusion_matrix(y_true, y_pred, labels = labels)
    
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = (cm.sum() - (FP + FN + TP))

    report['n_articles'] = n
    report['seed_no'] = seed
    report['params'] = params
    report['FP'] = FP.tolist()
    report['FN'] = FN.tolist()
    report['TP'] = TP.tolist()
    report['TN'] = TN.tolist()

    logger.warning(f"Model fit with {n} articles achieves model with following macro avg: {report.get('macro avg')}")

    # Write to file
    out_path = join(output_dir, "n-art_binary_model-eval_class-reps-1.json")

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

    return model

## FITTING MODELS
train_sizes = list(range(5,101,5)) # train size set as number of articles

for n_articles in train_sizes:
    fit_model(n_articles, seed_no, labels = labels)