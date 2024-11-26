#!/usr/bin/env python
# coding: utf-8

import os
from os.path import join
import pandas as pd
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset
from random import sample

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

mapdata_p = join(data_dir, 'ALL OUTCOMES AND MAPPING.xlsx')
modelinp = join(models_dir, 'musco_model')

# load model
model_trained = SetFitModel.from_pretrained(modelinp)

# Wrapper function
def predict_withproba(texts, model = model_trained, labels = ["Musculoskeletal and connective tissue outcomes", "Other"]):

    probas = model.predict_proba(texts).tolist()

    preds = []

    for c, probs in enumerate(probas):
        index_max = probs.index(max(probs))

        pred = {
            'text': texts[c],
            'label': labels[index_max],
            'prob': probs[index_max]
            }

        preds.append(pred)
    
    return(preds)

# Predict single
predict_withproba(["bapagsgsgowahgweg af afsa fsaf afwghreshe wag r gre gergre"])

# TEST ON DATA
## Load data
mapdf = pd.read_excel(mapdata_p, sheet_name=0) # mappings

## data handling
outcome_int = "Musculoskeletal and connective tissue outcomes"

cols_keep = ['Study-id', 'Verbatim Outcomes', 'Outcome Domains']
mapdf = mapdf.loc[:, cols_keep]
mapdf.loc[mapdf["Outcome Domains"] != outcome_int, "Outcome Domains"] = "Other"

## select sentences for testing
article_select = int(sample(list(mapdf['Study-id'].unique()), 1)[0])

data_test = mapdf.loc[mapdf['Study-id'] == article_select, :]
sentences_test = list(data_test.loc[:, 'Verbatim Outcomes'])

## predict
sentence_pred = predict_withproba(sentences_test)

pred_labels = [pred['label'] for pred in sentence_pred] # extract labels

data_test['pred_label'] = pred_labels # add to data

data_test['correct'] = data_test['Outcome Domains'] == data_test['pred_label'] # indicator for correct prediction

data_test['correct'].sum() / data_test['correct'].shape[0] # pct correct (accuraccy)


data_test[['Outcome Domains', 'pred_label']]