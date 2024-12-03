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

## LOGGING SETUP
logging.basicConfig(
    filename=join(logs_dir, 'predict_outcome-domains_nov24.log'),  # Log file name
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

predictdata_p = join(output_dir, 'outcome-notoutcome_predictions_nov24.csv')

## READ MAPPINGS WITH PREDICTIONS
predict_df = pd.read_csv(predictdata_p) # predictions

n_articles = len(predict_df.loc[predict_df['used_in_training'], 'Study ID'].unique().tolist())

### fix labels
labels_fix = {
    'Adverse_x000D_ events': 'Adverse events',
    'Blood and lymphatic system _x000D_outcomes': 'Blood and lymphatic system outcomes',
    'Infection and infestation\n outcomes': 'Infection and infestation outcomes'   
}

predict_df['label'] = predict_df['label'].replace(labels_fix)
predict_df['label'] = predict_df['label'].str.strip()

## FILTER FOR TRAINING
model_data = predict_df.loc[predict_df['used_in_training'], :] # use same articles as used in binary model

labels_exclude = ['Hospital', 'Need for further intervention', 'Economics'] # labels can be excluded (resource use outcomes)
model_data = model_data.loc[~model_data['label'].isin(labels_exclude), :] # exclude labels
model_data = model_data.loc[model_data['label'] != 'not outcome', :] # only keep verbatim outcomes

## IDS FOR MODELLING
ids_use = model_data['Study ID'].unique().tolist()

## SEED USED
seed_no = 4220

## TRAINING, EVAL DATA
random.seed(seed_no)
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
model_name = "thenlper/gte-base"
### labels
labels = model_data['label'].unique().tolist()
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


## MODEL TRAINING
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
y_true = list(eval_data['label'])
y_pred = model.predict(list(eval_data['text']))

report = classification_report(y_true, y_pred, output_dict = True, zero_division = 0)
cm = confusion_matrix(y_true, y_pred, labels = labels)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = (cm.sum() - (FP + FN + TP))

report['n_articles'] = n_articles
report['seed_no'] = seed_no
report['params'] = params
report['FP'] = FP.tolist()
report['FN'] = FN.tolist()
report['TP'] = TP.tolist()
report['TN'] = TN.tolist()

logger.warning(f"Model for outcome domains fit with {n_articles} articles achieves model with following macro avg: {report.get('macro avg')}")

# Write to file
out_path = join(output_dir, "report_outcome-domain-model_final_nov24.json")

with open(out_path, 'w') as f:
    json.dump(report, f)

# Save model
modeloutp = join(models_dir, 'od_model')
model.save_pretrained(modeloutp)

## PREDICTIONS
pred_outcomes_df = predict_df.loc[predict_df['predicted'] == 'outcome', :]

# predict
pred_outcomes_df['predicted_od'] = model.predict(list(pred_outcomes_df['text']))

# merge with predicted data
predict_combined_df = pd.merge(predict_df, pred_outcomes_df[['predicted_od']], how='left', left_index=True, right_index=True)

# write to file
predict_outp = join(output_dir, 'outcome-domains_predictions_nov24.csv')
predict_combined_df.to_csv(predict_outp, index = False)