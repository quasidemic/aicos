#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
import pandas as pd
from plotnine import ggplot, geom_line, aes, ggsave, scale_x_continuous, scale_y_continuous, theme_minimal
import json
import numpy as np

project_dir = join('/work', 'aicos')
output_dir = join(project_dir, 'output')
plot_dir = join(project_dir, 'plots')

## READ 
files_read = [file for file in os.listdir(output_dir) if file.startswith('n-art_binary_model')]

model_metrics_df = pd.DataFrame()

for file in files_read:
    fp = join(output_dir, file)

    with open(fp, 'r') as f:
        metrics = json.load(f)

    for m in metrics:
        m_select = {k:v for k,v in m.items() if k in ['outcome', 'not outcome', 'weighted avg', 'macro avg']}

        m_select_df = pd.DataFrame.from_dict(m_select, orient = 'columns').reset_index(names = 'metric')
        m_select_df = m_select_df.melt(id_vars='metric', value_vars=['outcome', 'not outcome', 'weighted avg', 'macro avg'], 
                                        var_name='target', value_name='score')
        m_select_df['n_articles'] = m.get('n_articles')
        m_select_df['accuracy'] = m.get('accuracy')
        m_select_df['FP_o'] = m.get('FP')[0]
        m_select_df['FP_no'] = m.get('FP')[1]
        m_select_df['FN_o'] = m.get('FN')[0]
        m_select_df['FN_no'] = m.get('FN')[1]
        m_select_df['TP_o'] = m.get('TP')[0]
        m_select_df['TP_no'] = m.get('TP')[1]
        m_select_df['TN_o'] = m.get('TN')[0]
        m_select_df['TN_no'] = m.get('TN')[1]

        model_metrics_df = pd.concat([model_metrics_df, m_select_df])



## PLOT
plot_df = model_metrics_df.loc[(model_metrics_df['target'] == 'outcome') & (model_metrics_df['metric'].isin(['precision', 'recall'])), :]

p = (
    ggplot(plot_df, aes("n_articles", "score", color = "metric"))
    + geom_line()
    + scale_x_continuous(breaks = list(range(5,105,5)))
    + scale_y_continuous(breaks = list(np.arange(0.0, 1.05, 0.05)))
    + theme_minimal()
    )

## SAVE
p.save(filename=join(plot_dir, 'bin-model_n-art_plot.png'), width=12, height=10, dpi=300)
model_metrics_df.to_csv(join(output_dir, 'metrics_binary-model_n-art.csv'), index = False)

## Conclusion: Use 30 based on recall

