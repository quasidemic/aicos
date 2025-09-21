#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
import pandas as pd
import random
from random import sample
import logging
import numpy as np
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score
import torch
import json

project_dir = join('/work', 'aicos')
modules_p = join(project_dir, 'modules')
logs_dir = join(project_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

sys.path.append(modules_p)

## DIRS AND PATHS
data_dir = join(project_dir, 'data')
plot_dir = join(project_dir, 'plots')
output_dir = join(project_dir, 'output')
models_dir = join(project_dir, 'models')

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

mapdata_p = join(data_dir, 'articles_outcomes_nov24.csv')
#negatives_p = join(data_dir, 'negatives.csv')
negatives_p = join(data_dir, 'article_negatives_nps.csv')

## READ MAPPINGS
mapdf = pd.read_csv(mapdata_p) # mappings

## READ NEGATIVES
negdf = pd.read_csv(negatives_p)

## DATA HANDLING
cols_keep = ['Study ID', 'Verbatim Outcomes', 'Outcome Domains']
has_results_filter_outcome = mapdf['has results'] == True
mapdf_model = mapdf.loc[has_results_filter_outcome, cols_keep].rename(columns={"Verbatim Outcomes": "text", "Outcome Domains": "label"})
mapdf_model['label'] = "outcome"

### rename negatives df columns
has_results_filter_negs = negdf['has results'] == True
negdf_model = negdf.loc[has_results_filter_negs,].rename(columns={"non outcome": "text"})
negdf_model['label'] = "not outcome"
negdf_model = negdf_model[['Study ID', 'text', 'label']]

## IDS FOR MODELLING
studyids = mapdf_model['Study ID'].unique().tolist()

## SEED USED FOR SAMPLING TRAINING, EVAL, TEST
seed_no = 4220

## SET FIXED TEST SET
random.seed(seed_no)

test_size = 0.25
test_ids = sample(studyids, round(test_size * len(studyids)))

# outcomes
test_df_outcome = mapdf_model.loc[mapdf_model['Study ID'].isin(test_ids), ].drop(columns = ['Study ID']) 
n_outcomes = test_df_outcome.shape[0]

# non outcomes
test_df_nonoutcome = negdf_model.loc[negdf_model['Study ID'].isin(test_ids), ].drop(columns = ['Study ID']) 
test_df_nonoutcome = test_df_nonoutcome.sample(n = n_outcomes, replace=False, random_state = seed_no, ignore_index = True)

test_df = pd.concat([test_df_outcome, test_df_nonoutcome], axis=0).reset_index(drop = True) # concatenate
test_data = Dataset.from_pandas(test_df, preserve_index = False)


## REMAINING IDS
train_eval_ids = [id for id in studyids if id not in test_ids]

## FUNCTION FOR TRAINING, EVAL DATA BASED ON N ARTICLES
def set_train_eval(n, seed, eval_prop = 0.2, train_eval_ids = train_eval_ids, outcome_data = mapdf_model, nonoutcome_data = negdf_model):

    # set seed
    random.seed(seed)
    
    # split ids
    ids_use = train_eval_ids[:n]

    train_prop = 1 - eval_prop
    train_ids = sample(ids_use, round(train_prop * len(ids_use)))
    eval_ids = list(set(ids_use) - set(train_ids))

    # set train, eval data based on ids
    # outcomes
    train_outcome_df = outcome_data.loc[outcome_data['Study ID'].isin(train_ids), ].drop(columns = ['Study ID'])
    n_train_outcomes = train_outcome_df.shape[0]
    eval_outcome_df = outcome_data.loc[outcome_data['Study ID'].isin(eval_ids), ].drop(columns = ['Study ID'])
    n_eval_outcomes = eval_outcome_df.shape[0]

    # non outcomes
    train_nonoutcome_df = nonoutcome_data.loc[nonoutcome_data['Study ID'].isin(train_ids), ].drop(columns = ['Study ID'])
    train_nonoutcome_df = train_nonoutcome_df.sample(n = n_train_outcomes, replace=False, random_state = seed, ignore_index = True)

    eval_nonoutcome_df = nonoutcome_data.loc[nonoutcome_data['Study ID'].isin(eval_ids), ].drop(columns = ['Study ID'])
    eval_nonoutcome_df = eval_nonoutcome_df.sample(n = n_eval_outcomes, replace=False, random_state = seed, ignore_index = True)

    # concatenate
    train_df = pd.concat([train_outcome_df, train_nonoutcome_df], axis=0).reset_index(drop = True) # concatenate
    eval_df = pd.concat([eval_outcome_df, eval_nonoutcome_df], axis=0).reset_index(drop = True)

    # convert to transformers dataset
    train_data = Dataset.from_pandas(train_df, preserve_index = False)
    eval_data = Dataset.from_pandas(eval_df, preserve_index = False)

    return ids_use, train_data, eval_data


## MODEL, LABELS
### model
model_name = join(models_dir, 'binary_model_nov24')
### labels
labels = ['outcome', 'not outcome']
### device
device = "cuda" if torch.cuda.is_available() else "cpu"


## SET TRAIN AND EVAL DATA 
n = 20 # use 20 based on 04_model-select-train
ids_use, train_data, eval_data = set_train_eval(n, seed_no)

# Load trained SetFit model
model = SetFitModel.from_pretrained(
    model_name,
    labels = labels
    ).to(device)

### parameters
params = {
    'head_params': {
        'solver': 'liblinear',
        'max_iter': 200
    },
    'batch_size': 64,
    'num_epochs': 2,
    'body_learning_rate': 1.5e-05
    }

# Evaluate
y_true = list(test_data['label'])
y_pred = model.predict(list(test_data['text']))
y_pred_proba = model.predict_proba(list(test_data['text']))[:, 1]

report = classification_report(y_true, y_pred, output_dict = True, zero_division = 0)
cm = confusion_matrix(y_true, y_pred, labels = labels)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = (cm.sum() - (FP + FN + TP))

report['AUROC'] = roc_auc_score(y_true, y_pred_proba)
report['AUPRC'] = average_precision_score(y_true, y_pred_proba, pos_label="outcome")
report['n_articles'] = n
report['seed_no'] = seed_no
report['params'] = params
report['FP'] = FP.tolist()
report['FN'] = FN.tolist()
report['TP'] = TP.tolist()
report['TN'] = TN.tolist()

# Write to file
out_path = join(output_dir, "report_binary-model_final_sep25.json")

with open(out_path, 'w') as f:
    json.dump(report, f)