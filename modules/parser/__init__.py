#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os.path import join
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from setfit import SetFitModel
import numpy as np
import pysbd
import re
import spacy

## DIRS AND PATHS
project_dir = join('/work', 'aicos')
data_dir = join(project_dir, 'data')
pdf_dir = join(data_dir, "articles")
plot_dir = join(project_dir, 'plots')
output_dir = join(project_dir, 'output')
models_dir = join(project_dir, 'models')

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Models
outcome_model_p = join(models_dir, 'binary_model')
od_model_p = join(models_dir, 'od_model')

## LOAD MODELS
outcome_model = SetFitModel.from_pretrained(outcome_model_p)
od_model = SetFitModel.from_pretrained(od_model_p)

# Load the English language model
nlp = spacy.load("en_core_web_sm")

## FUNCTION FOR FINDING SECTION CLOSEST TO "RESULTS"
def ident_results_section(sections, string_sect_start = 'results', string_sect_end = 'discussion', model_name='all-MiniLM-L6-v2'):

    # load model
    embeddings_model = SentenceTransformer(model_name)

    # section strings
    sections_s = [re.sub(r"[\n\t0-9\.]", "", s.group(0)).strip() for s in sections]

    # embed section strings
    sect_headings_embed = embeddings_model.encode(sections_s)

    # embed input strings
    sect_start_embed = embeddings_model.encode([string_sect_start])
    sect_end_embed = embeddings_model.encode([string_sect_end])

    # similarities
    simils_start = cosine_similarity(sect_start_embed, sect_headings_embed).flatten()
    simils_end = cosine_similarity(sect_end_embed, sect_headings_embed).flatten()

    # most similar embeddings (threshold?)
    match_start_score = np.max(simils_start)
    match_start_idx = np.argmax(simils_start)
    match_end_score = np.max(simils_end)
    match_end_idx = np.argmax(simils_end)

    # restrictions on article structure
    ## must contain results
    if match_start_score < 0.75:
        return None, None
    ## must contain discussion
    if match_end_score < 0.75:
        return None, None
    ## discussion must follow results
    if match_start_idx > match_end_idx:
        return None, None

    # return start, end
    return match_start_idx, match_end_idx

## FUNCTION FOR EXTRACT RESULTS SECTION
def extract_results(text):

    ## section regex
    sec_reg = re.compile(r"\n+(?:\b\S{1,3})?(?:[ \t]*\b[a-z]{3,}\b){1,5}[ \t]*\n+", re.IGNORECASE)
    
    # list of section headings based on regex
    sections = list(re.finditer(sec_reg, text))
    # filter sections based on span start (avoid results, discussion headings as part of abstract/introduction)
    len_t = len(text)
    span_thres = int(len_t/6)
    sections = [s for s in sections if s.start() > span_thres]

    # return if no sections
    if not sections:
        return

    # identify results-heading
    result_start, result_end = ident_results_section(sections)

    # Return if no result section or not valid structure
    if result_start is None or result_end is None:
        return # no result section found
        
    # Extract text from results
    start_pos = sections[result_start].start()
    end_pos = sections[result_end].start()

    text_results = text[start_pos:end_pos].strip()

    return text_results


## WRAPPER FUNCTION FOR PREDICT WITH PROBABILITY
def predict_withproba(texts, model):

    labels = sorted(model.labels) # predict_proba output matches sorted labels

    probas = model.predict_proba(texts).tolist()

    preds = []

    for c, probs in enumerate(probas):
        index_max = probs.index(max(probs))

        pred = {
            "s_id": c,
            "text": str(texts[c]),
            "label": labels[index_max],
            "prob": probs[index_max]
            }
        
        preds.append(pred)
    
    return preds

## FUNCTION FOR EXTRACTING NOUN PHRASES FROM SENTENCES
def extract_noun_phrases(text):

    # process the text with spaCy
    doc = nlp(text)

    # set to track token indices already included in a phrase
    covered_tokens = set()
    
    # list for noun phrases
    noun_phrases = []

    for chunk in doc.noun_chunks:

        # base phrase
        base_phrase = chunk.text
        
        # find root noun of the chunk
        root = chunk.root

        # check if this root or any token in the chunk has already been covered
        if any(token.i in covered_tokens for token in chunk):
            continue

        # list of adjectives directly modifying the root noun
        adjectives = [child.text for child in root.children if child.pos_ == "ADJ" and child.text not in str(chunk)]

        # list of prepositional phrases modifying the root noun
        prepositional_phrases = []
        for child in root.children:
            if child.dep_ == "prep":  # prepositional modifier
                # include the entire subtree of the preposition without duplicates
                prep_phrase = " ".join(descendant.text for descendant in child.subtree)
                prepositional_phrases.append(prep_phrase)
    
                # mark all tokens in the prepositional subtree as covered
                covered_tokens.update(descendant.i for descendant in child.subtree)
        
        # mark tokens in the current noun chunk as covered
        covered_tokens.update(token.i for token in chunk)

        # Build the full enhanced phrase
        full_phrase = " ".join(adjectives + [base_phrase] + prepositional_phrases)
        noun_phrases.append(full_phrase)

    return noun_phrases


## FUNCTION FOR SEGMENTING AND APPLYING MODEL
def segment_predict(results_text, segmenter=pysbd.Segmenter(language="en", clean=False), outcome_model=outcome_model, od_model=od_model):

    # clear linebreaks
    results_text = results_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # segment results to sentences
    sentences = segmenter.segment(results_text)

    # segment sentences to noun phrases
    results_phrases = []
    for sentence in sentences:
        sent_phrases = extract_noun_phrases(sentence)

        results_phrases.extend(sent_phrases)

    # remove duplicates
    results_phrases = list(set(results_phrases))

    # predict outcomes
    outcome_predictions = predict_withproba(results_phrases, outcome_model)

    # predicted verbatim outcomes
    verb_outc_pred = [pred for pred in outcome_predictions if pred.get("label") == "outcome"]
    verb_outc_pred_s = [pred.get("text") for pred in verb_outc_pred]

    if not verb_outc_pred: # return None if no verbatim outcomes
        return

    # predict outcome domain
    od_predictions_l = od_model.predict(verb_outc_pred_s).tolist()

    # add od
    for d, l in zip(verb_outc_pred, od_predictions_l):
        d["outcome_domain"] = l

    # return predictions
    return verb_outc_pred

## FUNCTION FOR PROCESSING PDF AND EXTRACTING OUTCOMES
def process_pdf(pdf_p):

    # filename
    file_name = os.path.basename(pdf_p)

    # extract text from pdf
    pdf_text = extract_text(pdf_p)

    # extract results text
    results_t = extract_results(pdf_text)

    # return if no results (does not meet requirements or results not found)
    if results_t is None:
        return "no results"
    
    # extract verbatim outcomes and OD
    verb_outcomes_preds = segment_predict(results_t)

    # return if no verbatim outcomes
    if not verb_outcomes_preds:
        return

    # add filename
    for d in verb_outcomes_preds:
        d["filename"] = file_name


    # return predictions
    return verb_outcomes_preds