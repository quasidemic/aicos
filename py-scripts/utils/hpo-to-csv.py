#!/usr/bin/env python
# coding: utf-8

import os
from os.path import join
import re
from ast import literal_eval
import pandas as pd

## SCRIPT FOR CONVERTING HPOS HYPERPARAMETER SEARCH TO CSV

output_dir = '/work/aicos/output'

hpo_files = [file for file in os.listdir(output_dir) if "hpo_trials" in file]

hpo_text = ''
for file in hpo_files:
    with open(join(output_dir, file), 'r') as f:
        file_string = f.read()
        hpo_text = hpo_text + '\n' + file_string

obj_re = re.compile(r'(?<=Objective value: )\d\.\d+')
hpos_re = re.compile(r'(?<=Hyperparameters: ).*?(?=\n|$)')

hpos = hpos_re.findall(hpo_text)
hpos = [literal_eval(hpo) for hpo in hpos]
objvs = obj_re.findall(hpo_text)

for i, hpo in enumerate(hpos):

    hpo['objective_value'] = float(objvs[i])

hpo_df = pd.DataFrame.from_records(hpos)

outpath = join(output_dir, 'aicos_hpo_nove24.csv')
hpo_df.to_csv(outpath, index = False)