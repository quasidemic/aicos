#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
import pandas as pd
from tqdm import tqdm

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

## PDF files
pdfs = os.listdir(pdf_dir)
pdfs = [filename for filename in pdfs if filename.endswith(".pdf")]

## PROCESS PDFs
print('processing pdfs...')
processed_pdfs = []
no_results = []
no_outcomes = []

for pdf in tqdm(pdfs):

    # full path to file
    pdf_path = join(pdf_dir, pdf)

    # process pdf
    pdf_outc_pred = process_pdf(pdf_path)

    if pdf_outc_pred == "no results":
        no_results.append(pdf)
        continue
    if pdf_outc_pred is None:
        no_outcomes.append(pdf)
        continue
    
    # append to list
    processed_pdfs.extend(pdf_outc_pred)

# printing missing results and outcomes
if no_results:
    print(f"No results section found for the following articles: {', '.join(no_results)}")
if no_outcomes:
    print(f"No outcomes found for the following articles: {', '.join(no_outcomes)}")

# convert to dataframe
print('converting to dataframe...')
pdfs_outc_preds_df = pd.DataFrame.from_records(processed_pdfs)

# rename columns
pdfs_outc_preds_df = pdfs_outc_preds_df.rename(columns={'text': 'verbatim_outcome', 'prob': 'is_outcome_probability'})
pdfs_outc_preds_df['article_id'] = pdfs_outc_preds_df['filename'].str.replace('.pdf', '')
pdfs_outc_preds_df = pdfs_outc_preds_df[['article_id', 'verbatim_outcome', 'sentence context', 'outcome_domain', 'is_outcome_probability']]

# export
print('exporting...')
out_p = join(output_dir, 'extracted_outcomes_20241129.csv')
pdfs_outc_preds_df.to_csv(out_p, index = False)

print('Done!')