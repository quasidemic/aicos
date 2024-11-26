#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json
from os.path import join

project_dir = join('/work', 'aicos')
output_dir = join(project_dir, 'output')

with open('/work/aicos/output/n-art_n-labels_model-eval_class-reps.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame()
skip = ['seed_no', 'params', 'FP', 'FN', 'TP', 'TN', 'labels']

for l in data:

    new_dict = {}

    for k in l:
        if k not in skip:
            new_dict[k] = l[k]

    l_df = pd.DataFrame.from_dict(new_dict, orient='columns').reset_index(names = 'metrics')

    not_labels = ['metrics', 'accuracy', 'macro avg', 'weighted avg', 'n_articles', 'n_labels']
    labels = l_df.columns[~l_df.columns.isin(not_labels)]

    #n_labels = len(labels[l_df.loc[l_df['metrics'] != 'support', labels].sum() > 0].tolist()) - 1 # deducting 1 for "no outcome" label

    #l_df['n_labels'] = n_labels

    df = pd.concat([df, l_df])

df.to_csv(join(output_dir, 'n-art_n-labels_model-eval.csv'), index = False)