#!/usr/bin/env python
# coding: utf-8

import os
import sys
from os.path import join
import pandas as pd
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import random
from random import sample
from optuna import Trial

project_dir = join('/work', 'aicos')
modules_p = join(project_dir, 'modules')
sys.path.append(modules_p)

from modelling import *

## DIRS AND PATHS
project_dir = join('/work', 'aicos')
data_dir = join(project_dir, 'data')
articles_dir = join(data_dir, 'articles')
plot_dir = join(project_dir, 'plots')
output_dir = join(project_dir, 'output')
models_dir = join(project_dir, 'models')

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

mapdata_p = join(data_dir, 'ALL OUTCOMES and ARTICLE ID and MAPPING FRAMEWORK_2024-08-22.xlsx')
modeloutp = join(models_dir, 'musco_model')

## READ MAPPINGS
mapdf = pd.read_excel(mapdata_p, sheet_name=0) # mappings
art_ids_df = pd.read_excel(mapdata_p, sheet_name=1) # id - article

## DATA HANDLING
outcome_int = "Musculoskeletal and connective tissue outcomes"
cols_keep = ['Study ID', 'Verbatim Outcomes', 'Outcome Domains']

mapdf_model = mapdf.loc[:, cols_keep].rename(columns={"Verbatim Outcomes": "text", "Outcome Domains": "label"})
mapdf_model.loc[mapdf_model["label"] != outcome_int, "label"] = "Other outcome"

studyids = mapdf['Study ID'].unique().tolist()

outcome_df = mapdf_model.loc[mapdf_model["label"] == outcome_int, :]
n_outcome = outcome_df.shape[0]
non_outcome_df = mapdf_model.loc[mapdf_model["label"] != outcome_int, :]
n_non_outcome = non_outcome_df.shape[0]

if n_outcome > n_non_outcome:
    replace_setting = True
else:
    replace_setting = False

non_outcome_df = non_outcome_df.sample(n = n_outcome, replace = replace_setting, random_state = 4220)

df_use = pd.concat([outcome_df, non_outcome_df])
ids_use = df_use['Study ID'].unique().tolist()

## ARTICLES FOR TRAIN, TEST, EVAL
train_size = 0.5
#train_size = 0.8

random.seed(4220)
train_ids = sample(ids_use, round(train_size * len(ids_use)))
test_eval_ids = list(set(ids_use) - set(train_ids))

## TRAIN, TEST DATA
train_df = df_use.loc[df_use['Study ID'].isin(train_ids), ].drop(columns = ['Study ID'])
test_eval_df = df_use.loc[df_use['Study ID'].isin(test_eval_ids), ].drop(columns = ['Study ID'])
test_df, eval_df = train_test_split(test_eval_df, test_size = 0.5, random_state = 22) # splitting temp to test eval (70% train, 15% test, 15% eval)

train_data = Dataset.from_pandas(train_df, preserve_index = False)
eval_data = Dataset.from_pandas(eval_df, preserve_index = False)
test_data = Dataset.from_pandas(test_df, preserve_index = False)
#test_data = Dataset.from_pandas(test_eval_df, preserve_index = False)

## MODEL AND LABELS
### model
model_name = "thenlper/gte-base"
### labels
labels = ["Musculoskeletal and connective tissue outcomes", "Other"]
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
        labels=labels,
        **params).to(device)

    return(model_use)

#### HP space
def hp_space(trial):
    return {
        "body_learning_rate": trial.suggest_float("body_learning_rate", 1e-6, 1e-2, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 2, 6),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 48, 64]),
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
    }

# Setfit Trainer
trainer = Trainer(
    train_dataset=train_data,
    eval_dataset=eval_data,
    model_init = model_init,
    metric=compute_metrics,
    column_mapping={"text": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
)

# Run HP opt
best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=10)

print(best_run)

trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
trainer.train()

metrics = trainer.evaluate(test_data)
print(metrics)


# Save model
trainer.model.save_pretrained(modeloutp)
