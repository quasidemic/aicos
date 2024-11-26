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
    filename=join(logs_dir, 'eval_train_size_20240930.log'),  # Log file name
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

### rename negatives df columns
negdf = negdf.rename(columns={"Verbatim Outcomes": "text", "Outcome Domains": "label"})

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
labels_filter = mapdf_model['label'].value_counts() > count_thres
labels_keep = list(mapdf_model['label'].value_counts()[labels_filter].index)

### filter labels
row_labels_filter = mapdf_model['label'].isin(labels_keep)
mapdf_model = mapdf_model[row_labels_filter]

### encode labels
#label_encoder = LabelEncoder()
#mapdf_model['label'] = label_encoder.fit_transform(mapdf_model['label'])

#labels_dict = dict(zip([int(l) for l in label_encoder.transform(labels_keep)], labels_keep))

## IDS FOR MODELLING
studyids = mapdf_model['Study ID'].unique().tolist()

## SEED USED FOR SAMPLING TRAINING, EVAL, TEST
seed_no = 4220

## SET FIXED TEST SET
random.seed(seed_no)

test_size = 0.25
test_ids = sample(studyids, round(test_size * len(studyids)))

test_df = mapdf_model.loc[mapdf_model['Study ID'].isin(test_ids), ]

## REMAINING IDS
train_eval_ids = [id for id in studyids if id not in test_ids]
shuffle(train_eval_ids)

## FUNCTION FOR TRAINING, EVAL DATA BASED ON N ARTICLES
def set_train_eval(n, seed, labels, eval_prop = 0.2, train_eval_ids = train_eval_ids, df_use = mapdf_model, negdf_use = negdf):

    # set seed
    random.seed(seed)
    
    # split ids
    ids_use = train_eval_ids[:n]

    train_prop = 1 - eval_prop
    train_ids = sample(ids_use, round(train_prop * len(ids_use)))
    eval_ids = list(set(ids_use) - set(train_ids))

    # set train data based on ids
    train_df = df_use.loc[(df_use['Study ID'].isin(train_ids)) & (df_use['label'].isin(labels)), ]

    # add negatives
    negdf_f = negdf_use.loc[negdf_use['Study ID'].isin(train_ids), :]
    n_negs_add = round(train_df.shape[0] / len(labels))
    if n_negs_add > negdf_f.shape[0]:
        n_negs_add = negdf_f.shape[0]
    neg_sample = negdf_f.sample(n = n_negs_add)
    train_df = pd.concat([train_df, neg_sample]).drop(columns = ['Study ID']).reset_index(drop = True)

    # Get labels
    #labels = train_df['label'].unique().tolist()

    # set eval data based on ids and labels
    eval_df = df_use.loc[(df_use['Study ID'].isin(eval_ids)) & (df_use['label'].isin(labels)), ]

    # add negatives
    negdf_f = negdf_use.loc[negdf_use['Study ID'].isin(eval_ids), :]
    n_negs_add = round(eval_df.shape[0] / len(labels))
    if n_negs_add > negdf_f.shape[0]:
        n_negs_add = negdf_f.shape[0]
    neg_sample = negdf_f.sample(n = n_negs_add)
    eval_df = pd.concat([eval_df, neg_sample]).drop(columns = ['Study ID']).reset_index(drop = True)
    
    # convert to transformers dataset
    train_data = Dataset.from_pandas(train_df, preserve_index = False)
    eval_data = Dataset.from_pandas(eval_df, preserve_index = False)

    return train_data, eval_data


## FUNCTION FOR OUT-OF-CLASS PREDICTION
def predict_ooc(model, texts, threshold = 0.70):

    labels = sorted(model.labels) # predict_proba output matches sorted labels
    probas = model.predict_proba(texts).tolist()

    preds = []

    for probs in (probas):

        max_prop = max(probs)

        index_max = probs.index(max_prop)

        if max_prop < threshold:
            label_out = 'No outcome'
        else:
            label_out = labels[index_max]

        preds.append(label_out)
    
    return(preds)


## MODEL, LABELS AND HYPERPARAMETERS
### model
model_name = "thenlper/gte-base"
### labels
labels = labels_keep
### device
device = "cuda" if torch.cuda.is_available() else "cpu"

### parameters from HPO
params = {
    'head_params': {
        'solver': 'liblinear',
        'max_iter': 100
    },
    'batch_size': 64,
    'num_epochs': 2,
    'body_learning_rate': 5e-06
    }

## FUNCTION FOR MODELLING BASED ON N ARTICLES
def fit_model(n, seed, labels, model_name = model_name, params = params, device = device, test_df = test_df, negdf = negdf, logger = logger):

    logger.warning(f"Fitting model using {len(labels)} labels and {n} articles...")

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
        load_best_model_at_end=True
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
    
    # Filter test data
    test_df = test_df.loc[test_df['label'].isin(labels), :]
    negdf_f = negdf.loc[negdf['Study ID'].isin(test_df['Study ID'].unique().tolist()), :]

    # Add negatives
    n_negs_add = round(test_df.shape[0] / len(labels))
    neg_sample = negdf_f.sample(n = n_negs_add)
    test_df_m = pd.concat([test_df, neg_sample]).drop(columns = ['Study ID']).reset_index(drop = True)

    # Conver to Dataset
    test_data = Dataset.from_pandas(test_df_m, preserve_index = False)

    # Evaluate
    y_true = list(test_data['label'])
    #y_pred = predict_ooc(model, list(test_data['text']), threshold = 0.6)
    y_pred = model.predict(list(test_data['text']))

    report = classification_report(y_true, y_pred, output_dict = True, zero_division = 0)
    cm = confusion_matrix(y_true, y_pred, labels = labels + ["No outcome"])
    
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = (cm.sum() - (FP + FN + TP))

    report['n_articles'] = n
    report['seed_no'] = seed
    report['params'] = params
    report['n_labels'] = len(labels)
    report['labels'] = labels
    report['FP'] = FP.tolist()
    report['FN'] = FN.tolist()
    report['TP'] = TP.tolist()
    report['TN'] = TN.tolist()

    logger.warning(f"Model fit with {len(labels)} labels and {n} articles achieves model with following macro avg: {report.get('macro avg')}")

    # Write to file
    out_path = join(output_dir, "n-art_n-labels_model-eval_class-reps.json")

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


## DETERMINE MOST COMMON LABELS (from test_df)
labels_exclude = ['Hospital', 'Need for further intervention', 'Economics'] # labels can be excluded (resource use outcomes)
labels_ordered = list(test_df['label'].value_counts().index)
labels_ordered = [label for label in labels_ordered if not label in labels_exclude]


## FITTING MODELS
train_sizes = list(range(5,101,5)) # train size set as number of articles
n_labels_iter = [1, 2, 3, 4, 5, 6]

for n_labels in n_labels_iter:

    labels_use = labels_ordered[:n_labels]

    for n_articles in train_sizes:
        fit_model(n_articles, seed_no, labels_use)