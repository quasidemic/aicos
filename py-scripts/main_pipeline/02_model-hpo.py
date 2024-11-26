#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
import pandas as pd
import re
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import random
from random import sample
from optuna import Trial
import torch
import logging

project_dir = join('/work', 'aicos')
modules_p = join(project_dir, 'modules')
logs_dir = join(project_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)

sys.path.append(modules_p)

from modelling import *

## LOGGING SETUP
logging.basicConfig(
    filename=join(logs_dir, 'hpo_binary-model.log'),  # Log file name
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
def set_train_eval(n, seed, eval_prop = 0.2, train_eval_ids = train_eval_ids, df_use = model_data):

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
train_data, eval_data = set_train_eval(25, seed_no, eval_prop = 0)

## MODEL AND LABELS
### model
model_name = "thenlper/gte-base"
### labels
labels = ['outcome', 'not outcome']
### device
device = "cuda" if torch.cuda.is_available() else "cpu"

## HYPERPARAMETER OPTIMIZATION
### Functions for HP-opt
#### model init
def model_init(params):
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }

    model_use = SetFitModel.from_pretrained(
        model_name,
        labels = labels,
        **params).to(device)

    return(model_use)

#### HP space
def hp_space(trial):
    return {
        #"body_learning_rate": trial.suggest_float("body_learning_rate", 1e-7, 1e-5, log=True), # first run
        "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-7, 1e-4, log=True), # second run
        "num_epochs": trial.suggest_int("num_epochs", 2, 6),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 48, 64]),
        "max_iter": trial.suggest_categorical("max_iter", [50, 100, 150, 200, 250, 300]),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
    }

# Setfit Trainer
trainer = Trainer(
    train_dataset = train_data,
    eval_dataset = test_data,
    model_init = model_init,
    metric = "accuracy",
    column_mapping={"text": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
)

# Run HP opt
best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=50)

for trial in best_run.backend.trials:
    logger.warning(f"Trial {trial.number} | Objective value: {trial.value} | Hyperparameters: {trial.params}")

logger.warning(best_run)
