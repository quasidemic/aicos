#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
from pdfminer.high_level import extract_text
import pandas as pd
import pysbd
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

## DIRS AND PATHS
project_dir = join('/work', 'aicos')
data_dir = join(project_dir, 'data')
pdf_dir = join(data_dir, "extracted_files", "AI in COS")
plot_dir = join(project_dir, 'plots')
output_dir = join(project_dir, 'output')
models_dir = join(project_dir, 'models')

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

mapdata_p = join(data_dir, 'ALL OUTCOMES and ARTICLE ID and MAPPING FRAMEWORK_2024-08-22.xlsx')

## EMBEDDING MODEL
model = SentenceTransformer('thenlper/gte-base')

## PDF files
pdfs = os.listdir(pdf_dir)
pdfs = [filename for filename in pdfs if filename.endswith(".pdf")]

## READ PDFS
articles = {}

for pdf in pdfs:
    pdf_path = join(pdf_dir, pdf)

    text = extract_text(pdf_path)

    if "case report" in text[0:1000].lower():
        case_report = 1
    else:
        case_report = 0

    entry = {
        'text': text,
        'is_case_report': case_report
    }

    key = pdf.replace('.pdf', '')

    articles[key] = entry


## READ MAPPINGS
mapdf = pd.read_excel(mapdata_p, sheet_name=0) # mappings
art_ids_df = pd.read_excel(mapdata_p, sheet_name=1) # id - article


## segmenter
seg = pysbd.Segmenter(language="en", clean=False)


## function for getting candidate negatives (sentences from sections prior to results)
def get_negs_cand(article_key, articles = articles):

    article = articles.get(article_key)

    if article is None:
        return

    if bool(article.get('is_case_report')):
        return

    text = article.get('text')

    text = text.replace("\nr e s u lts\n", "\nresults\n")
    text = text.replace('\x0c', '')

    # Regex pattern to find all headings (assuming headings are surrounded by line breaks)
    sec_reg = re.compile(r"\n+\b\S{1,}(?:[ \t]*\b\w{3,}\b){0,5}[ \t]*\n+")
    
    # Compile the heading pattern
    sections = list(re.finditer(sec_reg, text))

    # Search for the heading containing the word "result"
    result_heading_idx = None
    for idx, match in enumerate(sections):
        if re.search(r'result', match.group(0), re.IGNORECASE):
            result_heading_idx = idx
            break
    
    # Return if no result section
    if result_heading_idx is None:
        return # no result section found

    # Extract text leading up to results-section
    start_pos = sections[0].start()
    end_pos = sections[result_heading_idx].start()

    text_preresults = text[start_pos:end_pos]

    # Sentences
    neg_cands = seg.segment(text_preresults)
    neg_cands = [sentence for sentence in neg_cands if len(sentence) > 25]

    return neg_cands


## function for filtering negatives based on cosine similarity of verbatim outcomes
def filter_negs(article_key, articles = articles, mapdf = mapdf, model = model):

    # get potental negatives
    neg_cands = get_negs_cand(article_key)

    if neg_cands is None or len(neg_cands) == 0:
        return

    # extract verbatim outcomes
    verbats = mapdf.loc[mapdf['Study ID'] == int(article_key), 'Verbatim Outcomes'].to_list()
    n_verbats = len(verbats)

    # embed sentences
    embeddings_negs = model.encode(neg_cands)
    embeddings_verbats = model.encode(verbats)

    dists = []

    for neg_embed in embeddings_negs:

        dist = sum([cos_sim(neg_embed, verbat_embed) for verbat_embed in embeddings_verbats])

        dists.append(dist)

    negs_dist = dict(zip(neg_cands, dists))

    dists_sorted = dists.copy()
    dists_sorted.sort()

    if n_verbats > len(dists):
        n_get = len(dists)-1
    else:
        n_get = n_verbats

    threshold = dists_sorted[n_get]

    negs_keep = [neg_cand for neg_cand, dist in negs_dist.items() if dist <= threshold]

    return(negs_keep)
    

# dataframe with negs
negs_df = pd.DataFrame()

for article_key in articles:

    negs = filter_negs(article_key)

    if negs is None:
        continue

    study_negs_df = pd.DataFrame({
        'Study ID': int(article_key),
        'Verbatim Outcomes': negs,
        'Outcome Domains': 'No outcome'
        }
        )

    negs_df = pd.concat([negs_df, study_negs_df], axis = 0)


# export
negs_df.to_csv(join(data_dir, 'negatives.csv'), index = False)


# add case report indicator to data
articles_df = pd.DataFrame.from_dict(articles, orient='index').reset_index(names = 'Study ID')
articles_df['Study ID'] = articles_df['Study ID'].astype(int)

# join
all_data_df = pd.merge(mapdf, articles_df, on = 'Study ID')

# export
cols_keep = ['Study ID', 'Verbatim Outcomes', 'Outcome Domains', 'Core Areas', 'text', 'is_case_report']
all_data_df = all_data_df[cols_keep]
all_data_df.to_csv(join(data_dir, 'articles_text_outcomes.csv'), index = False)