#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
import pandas as pd
import json
import numpy as np
import json

project_dir = join('/work', 'aicos')
output_dir = join(project_dir, 'output')
plot_dir = join(project_dir, 'plots')

## READ 
cv_mets_p = join(output_dir, 'cross-val_binary_model_nov24.json')
od_mets_p = join(output_dir, 'rreport_outcome-domain-model_final_nov24.json')
bm_mets_p = join(output_dir, 'report_binary-model_final_nov24.json')

## CV
with open(cv_mets_p, 'r') as f:
    cv_metrics = json.load(f)

model_metrics_df = pd.DataFrame()

for k, m in enumerate(cv_metrics, start=1):
    m_select = {k:v for k,v in m.items() if k in ['outcome', 'not outcome', 'weighted avg', 'macro avg']}

    m_select_df = pd.DataFrame.from_dict(m_select, orient = 'columns').reset_index(names = 'metric')
    m_select_df = m_select_df.melt(id_vars='metric', value_vars=['outcome', 'not outcome', 'weighted avg', 'macro avg'], 
                                    var_name='target', value_name='score')
    m_select_df['n_articles'] = m.get('n_articles')
    m_select_df['accuracy'] = m.get('accuracy')
    m_select_df['fold'] = k

    model_metrics_df = pd.concat([model_metrics_df, m_select_df])

cv_mets_out = join(output_dir, 'cross-val_bm_metrics_nov24.csv')
model_metrics_df.to_csv(cv_mets_out, index=False)

## BM
with open(bm_mets_p, 'r') as f:
    bm_metrics = json.load(f)

m_select = {k:v for k,v in bm_metrics.items() if k in ['outcome', 'not outcome', 'weighted avg', 'macro avg']}

m_select_df = pd.DataFrame.from_dict(m_select, orient = 'columns').reset_index(names = 'metric')
m_select_df = m_select_df.melt(id_vars='metric', value_vars=['outcome', 'not outcome', 'weighted avg', 'macro avg'], 
                                var_name='target', value_name='score')

bm_mets_out = join(output_dir, 'bm_metrics_nov24.csv')
m_select_df.to_csv(bm_mets_out, index=False)

## FP, FN, TP, TN
conf_mat_outc = pd.DataFrame(
    {
        'Predicted': ['True', 'True', 'False', 'False'],
        'Actual': ['True', 'False', 'True', 'False'],
        'count': [bm_metrics.get('TP')[0], bm_metrics.get('FP')[0], bm_metrics.get('FN')[0], bm_metrics.get('TN')[0]]

})

conf_mat_out = join(output_dir, 'bm_conf_mat_nov24.csv')
conf_mat_outc.to_csv(conf_mat_out, index=False)

## OD model
with open(od_mets_p, 'r') as f:
    od_metrics = json.load(f)

m_select = {k:v for k,v in od_metrics.items() if k in ['weighted avg', 'macro avg']}

m_select_df = pd.DataFrame.from_dict(m_select, orient = 'columns').reset_index(names = 'metric')
m_select_df = m_select_df.loc[m_select_df['metric'].isin(['precision', 'recall', 'f1-score']), :]
m_select_df = m_select_df.transpose().reset_index().rename(columns = {'index': 'eval', 0: 'precision', 1: 'recall', 2: 'f1-score'})
m_select_df = m_select_df.iloc[1:, ]

od_mets_out = join(output_dir, 'odm_metrics_nov24.csv')
m_select_df.to_csv(od_mets_out, index=False)