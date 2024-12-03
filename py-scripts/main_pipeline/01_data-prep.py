#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
from pdfminer.high_level import extract_text
import pandas as pd
import re

project_dir = join('/work', 'aicos')
modules_dir = join(project_dir, 'modules')

sys.path.append(modules_dir)

from parser import *

## DIRS AND PATHS
data_dir = join(project_dir, 'data')
pdf_dir = join(data_dir, 'articles_2024-08-22')
plot_dir = join(project_dir, 'plots')
output_dir = join(project_dir, 'output')
models_dir = join(project_dir, 'models')

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

mapdata_p = join(data_dir, 'ALL OUTCOMES and ARTICLE ID and MAPPING FRAMEWORK_2024-08-22.xlsx')
negatives_p = join(data_dir, 'article_negatives_nps.csv')

## PDF files
pdfs = os.listdir(pdf_dir)
pdfs = [filename for filename in pdfs if filename.endswith(".pdf")]

## READ MAPPINGS
mapdf = pd.read_excel(mapdata_p, sheet_name=0) # mappings
art_ids_df = pd.read_excel(mapdata_p, sheet_name=1) # id - article

## READ NEGATIVES
negdf = pd.read_csv(negatives_p)

# add indicator whether article has negatives (and therefore used in training, extracting etc.)
art_indic = negdf[['Study ID', 'has results', 'has negatives']].drop_duplicates()

# join
all_data_df = pd.merge(mapdf, art_indic, on = 'Study ID', how='left')

# export
cols_keep = ['Study ID', 'Verbatim Outcomes', 'Outcome Domains', 'Core Areas', 'has results', 'has negatives']
all_data_df = all_data_df[cols_keep]
all_data_df.to_csv(join(data_dir, 'articles_outcomes_nov24.csv'), index = False)