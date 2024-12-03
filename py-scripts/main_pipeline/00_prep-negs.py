#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import pysbd

project_dir = join('/work', 'aicos')
modules_dir = join(project_dir, 'modules')

sys.path.append(modules_dir)

from parser import extract_noun_phrases, extract_results, extract_preresults

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

## PDF files
pdfs = os.listdir(pdf_dir)
pdfs = [filename for filename in pdfs if filename.endswith(".pdf")]

## READ MAPPINGS
mapdf = pd.read_excel(mapdata_p, sheet_name=0) # mappings
art_ids_df = pd.read_excel(mapdata_p, sheet_name=1) # id - article

## LOAD TRANSFORMER
stmodel = SentenceTransformer('thenlper/gte-base')
#stmodel = SentenceTransformer('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')

## CALCULATE DISTANCE THRESHOLD (BASED ON DISTANCES WITHIN VERBATIM OUTCOMES)
# verbatim outcomes as list
verbats = mapdf['Verbatim Outcomes'].tolist()

# embeddings
verbats_embeds = stmodel.encode(verbats)

# cosine similarities (distances)
cossims = cosine_similarity(verbats_embeds)

# flatten (upper diagonal)
cossims_flat = cossims[np.triu_indices_from(cossims, k=1)]

# minimum
MIN_DIST = np.min(cossims_flat)

# mean
MEDIAN_DIST = np.median(cossims_flat)

## FUNCTION FOR EXTRACTING NEGATIVES
def get_negs2(text, embeds_compare=verbats_embeds, threshold=0.85, segmenter=pysbd.Segmenter(language="en", clean=False), model = stmodel):

    # extract preresults
    preresults_text = extract_preresults(text)

    # segment results to sentences
    sentences = segmenter.segment(preresults_text)

    # segment sentences to noun phrases
    neg_cands = []
    for sentence in sentences:
        sent_phrases = extract_noun_phrases(sentence)

        neg_cands.extend(sent_phrases)

    # remove duplicates
    neg_cands = list(set(neg_cands))

    # return if no phrases found
    if not neg_cands:
        return

    # embed sentences
    embeddings_negs = model.encode(neg_cands)

    # distances
    negs_embed_zipped = dict(zip(neg_cands, embeddings_negs))

    negs_keep = []

    for neg, embed in negs_embed_zipped.items():

        dists = [cos_sim(embed, verbat_embed) for verbat_embed in embeds_compare]

        max_cossim = np.max(dists)

        if max_cossim > threshold:
            continue
        else:
            negs_keep.append(neg)

    return negs_keep


## FUNCTION FOR PROCESSING PDF FOR NEGATIVES (NOT OUTCOME)
def process_pdf_negs(pdf_p):

    # filename
    pdf_name = os.path.basename(pdf_p)

    # article id
    article_id = pdf_name.replace('.pdf', '')

    # extract text
    text = extract_text(pdf_p)

    # extract results text
    results_t = extract_results(text)

    # has results indicator (does not meet requirements or results not found)
    if results_t is None:
        has_results = False
    else:
        has_results = True

    # return empty df
    if not has_results:
        study_negs_df = pd.DataFrame({
        'Study ID': int(article_id),
        'non outcome': [''],
        'has results': has_results,
        'has negatives': False
        }
        )

        return

    # extract negatives
    negs = get_negs2(text, verbats_embeds)

    # has negs indicator
    if negs:
        has_negs = True
    else:
        has_negs = False
        negs = ['']

    # negs as df
    study_negs_df = pd.DataFrame({
        'Study ID': int(article_id),
        'non outcome': negs,
        'has results': has_results,
        'has negatives': has_negs
        }
        )

    return study_negs_df


## PROCESS PDFS FOR NEGS AND COMPILE TO DF
negs_df = pd.DataFrame()

for pdf in pdfs:

    # full path to pdf
    pdf_p = join(pdf_dir, pdf)

    # extract negs
    pdf_negs_df = process_pdf_negs(pdf_p)

    # concat to df
    negs_df = pd.concat([negs_df, pdf_negs_df], axis = 0)

## EXPORT
negs_df.to_csv(join(data_dir, 'article_negatives_nps.csv'), index = False)