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
from sklearn.metrics import classification_report, confusion_matrix
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
predictdata_p = join(output_dir, 'outcome-notoutcome_predictions_nov24.csv')

## READ MAPPINGS
mapdf = pd.read_csv(mapdata_p) # mappings

## READ MAPPINGS WITH PREDICTIONS
predict_df = pd.read_csv(predictdata_p) # predictions

n_articles = len(predict_df.loc[predict_df['used_in_training'], 'Study ID'].unique().tolist())


## DATA HANDLING
cols_keep = ['Study ID', 'Verbatim Outcomes', 'Outcome Domains']
has_results_filter = mapdf['has results'] == True
mapdf_model = mapdf.loc[has_results_filter, cols_keep].rename(columns={"Verbatim Outcomes": "text", "Outcome Domains": "label"})

### fix labels
labels_fix = {
    'Adverse_x000D_ events': 'Adverse events',
    'Blood and lymphatic system _x000D_outcomes': 'Blood and lymphatic system outcomes',
    'Infection and infestation\n outcomes': 'Infection and infestation outcomes'   
}

predict_df['label'] = predict_df['label'].replace(labels_fix)
predict_df['label'] = predict_df['label'].str.strip()

mapdf_model['label'] = mapdf_model['label'].replace(labels_fix)
mapdf_model['label'] = mapdf_model['label'].str.strip()

## FILTER FOR TRAINING
model_data = predict_df.loc[predict_df['used_in_training'], :] # use same articles as used in binary model

labels_exclude = ['Hospital', 'Need for further intervention', 'Economics'] # labels can be excluded (resource use outcomes)
model_data = model_data.loc[~model_data['label'].isin(labels_exclude), :] # exclude labels
model_data = model_data.loc[model_data['label'] != 'not outcome', :] # only keep verbatim outcomes

mapdf_model = mapdf_model.loc[~mapdf_model['label'].isin(labels_exclude), :] # exclude labels

## ALL IDS
studyids = mapdf_model['Study ID'].unique().tolist()

## IDS USED FOR BINARY MODEL
ids_use = model_data['Study ID'].unique().tolist()

## SEED USED
seed_no = 4220

## SET FIXED TEST SET
random.seed(seed_no)

test_size = 0.25
test_ids = sample(studyids, round(test_size * len(studyids)))

test_df = mapdf_model.loc[mapdf_model['Study ID'].isin(test_ids), ].drop(columns = ['Study ID']) 

test_data = Dataset.from_pandas(test_df, preserve_index = False)

## TRAINING, EVAL DATA
eval_prop = 0.2
train_prop = 1 - eval_prop

train_ids = sample(ids_use, round(train_prop * len(ids_use)))
eval_ids = list(set(ids_use) - set(train_ids))

# set train data based on ids
train_df = model_data.loc[(model_data['Study ID'].isin(train_ids)), ].drop(columns = ['Study ID'])

# set eval data based on ids and labels
eval_df = model_data.loc[(model_data['Study ID'].isin(eval_ids)), ].drop(columns = ['Study ID'])

# convert to transformers dataset
train_data = Dataset.from_pandas(train_df, preserve_index = False)
eval_data = Dataset.from_pandas(eval_df, preserve_index = False)


## MODEL, LABELS AND HYPERPARAMETERS
### model
model_name = join(models_dir, 'od_model_nov24')
### labels
labels = model_data['label'].unique().tolist()
### device
device = "cuda" if torch.cuda.is_available() else "cpu"

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
y_pred_proba = model.predict_proba(list(test_data['text']))

report = classification_report(y_true, y_pred, output_dict = True, zero_division = 0)
cm = confusion_matrix(y_true, y_pred, labels = labels)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = (cm.sum() - (FP + FN + TP))

report['AUROC'] = roc_auc_score(y_true, y_pred_proba, multi_class = "ovo", average = "weighted", labels = sorted(labels))
# AUPRC not possible for multi-class
report['n_articles'] = n_articles
report['seed_no'] = seed_no
report['params'] = params
report['FP'] = FP.tolist()
report['FN'] = FN.tolist()
report['TP'] = TP.tolist()
report['TN'] = TN.tolist()

# Write to file
out_path = join(output_dir, "report_outcome-domain-model_final_sep25.json")

with open(out_path, 'w') as f:
    json.dump(report, f)