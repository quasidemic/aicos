# **Applying** Machine Learning to Enhance Core Outcome Set Development

This repository contains a complete end-to-end pipeline for building, tuning, evaluating, and applying text classification models that identify *"outcome" vs "not outcome"* statements in medical articles (PDFs) and *categorize outcomes into COMET outcome domains*. The pipeline is implemented in modular Python scripts (located in `py-scripts`) that run in a specific order (as indicated by the numeric prefixes in filenames), with custom functions for parsing and modelling located in the `modules` directory. 

Furthermore, the repository contains a program that unifies pre-trained models to a single script capable of 1) parsing the text layer of a PDF, (2) identifying the results section, (3) extracting noun phrases from this section, (4) applying an outcome classification model to identify verbatim outcomes, and (5) using an outcome classification model to assign each outcome to a predefined COMET domain.

The code was used in the forthcoming research article: Yalcinkaya, A; Kjelmann, K G; Gholinezhad, S; Rahbek, O; Kold, S & Husum, H-C 2025: *Applying Machine Learning to Enhance Core Outcome Set Development: Automating the Data Extraction and Classification of Outcomes*

This README is structured as a *walkthrough* of the pipeline stages, explaining what each script does and how they work together to process data and train models, followed by a description of the unified script/program.

------

## Pipeline steps

### Overview

The pipeline consists of:

1. **Negative Example Generation**: Automatic mining of "not outcome" phrases from PDFs
2. **Data Preparation**: Extract and prepare labeled text data
3. **Hyperparameter Optimization**: Deriving optimal classifier settings
4. **Training Size Experiment**: Assessing model performance vs training size based on number of articles
5. **Model Selection & Training**: Choosing final training set size and model
6. **Cross-Validation**: Evaluation of selected model for classiciation of outcome phrases
7. **Outcome Domain Model**: Training and evaluating a multiclass classifier for assigning COMET outcome domains "outcome" text.
8. **Postprocessing**: Collating all metrics and generating output reports.

------

### Pipeline Steps

### 0️⃣ `00_prep-negs.py`

**Purpose**: Extract "not outcome" (negative) training examples from PDFs. The negatives provide balanced training data for the binary classifier.

- Reads full-text PDFs of studies.
- Identifies the "Results" and "Discussion" section of the studies.
- Segments noun phrases from sections other than "Results" and "Discussion" into candidate phrases and filters them based on semantic similarity with known outcome phrases ("Verbatim Outcomes").
- Outputs a CSV of "non outcome" phrases with metadata about their extraction.

------

### 1️⃣ `01_data-prep.py`

**Purpose**: Consolidate and align the main dataset for training, ensuring all articles have harmonized labels and metadata for modeling. 

- Merges outcome mapping framework with the generated negative samples.
- Cleans labels (fixes naming inconsistencies).
- Excludes unwanted labels (resource use outcomes).
- Produces a unified CSV containing both outcome and non-outcome examples with consistent columns.

------

### 2️⃣ `02_model-hpo_1.py`

**Purpose**: Perform *hyperparameter optimization (HPO)* for the binary classifier ("outcome" vs "not outcome"). This ensures best use of limited data by exploring learning rates, solver types, batch sizes, and epochs. 

- Utilizes the SetFit framework to train a model via few-shot learning with the [`thenlper/gte-base`](https://huggingface.co/thenlper/gte-base) model as the text encoder.
- Defines a *search space* for hyperparameters using Optuna (via SetFit):
  - `body_learning_rate`: log-uniform between 1e-7 and 1e-4
  - `num_epochs`: integer between 2 and 6
  - `batch_size`: categorical [16, 32, 48, 64]
  - `max_iter`: categorical [50, 100, 150, 200, 250, 300]
  - `solver`: categorical [lbfgs, liblinear]
- Uses *TPESampler* to explore this space with 50 trials (5 identical scripts with 10 trials each in order to parallelize run).
- Each trial initializes a SetFit model with the proposed hyperparameters.
- Trains on a fixed training/eval split and evaluates on a *fixed test set*.
- Logs the best trial and all hyperparameters tested.

------

### 3️⃣ `03_model-train-size_X-Y.py`

**Purpose**: Study *impact of training set size* on model performance. This experiment quantifies how many labeled articles are needed for good precision/recall. 

- Like step 2, utilizes the SetFit framework to train a model via few-shot learning with the [`thenlper/gte-base`](https://huggingface.co/thenlper/gte-base) model as the text encoder.
- Uses hyperparameters selected based on HPO results:
  - `head_params`: `{'solver': 'liblinear', 'max_iter': 200}`
  - `batch_size`: 64
  - `num_epochs`: 2
  - `body_learning_rate`: 1.5e-5
- Note: The hyperparameters are not the ones that achieved *the* best performance in the HP search, but they achieve almost the same performance but with a larger batch size and fewer number of epochs, reducing computation time substantially.
- For each training size from 5 to 100, increasing in steps of 5:
  - Randomly samples that many articles.
  - Builds balanced train/eval splits (outcome vs non-outcome).
  - Trains and evaluates with *same hyperparameters*.
  - Tests against the same fixed hold-out set of 28 articles.
  - Logs classification metrics and confusion matrices.
- Note that this step is split in five script to ease parallelization.

------

### 4️⃣ `04_model-select-train.py`

**Purpose**: Choose final training set size (20 articles) and *fit binary model to be used in unified script/program*.

- Like step 2 and 3, utilizes the SetFit framework to train a model via few-shot learning with the [`thenlper/gte-base`](https://huggingface.co/thenlper/gte-base) model as the text encoder.
- Uses same hyperparameters as in step 3 based on step 2 trials:
  - `head_params`: `{'solver': 'liblinear', 'max_iter': 200}`
  - `batch_size`: 64
  - `num_epochs`: 2
  - `body_learning_rate`: 1.5e-5
- Uses balanced training/eval splits of 20 articles (outcome / not outcome).
- Trains SetFit model and saves:
  - Final predictions on all study text.
  - Model artifact for reuse.
  - Classification report and confusion matrix.

------

### 5️⃣ `05_model-outcome_cross-val.py`

**Purpose**: Perform *5-fold cross-validation* of binary model. This step verifies robustness of the training set size across splits. 

- Like step 2-4, utilizes the SetFit framework to train a model via few-shot learning with the [`thenlper/gte-base`](https://huggingface.co/thenlper/gte-base) model as the text encoder.
- Uses same hyperparameters as in step 3 and 4 based on step 2 trials:
  - `head_params`: `{'solver': 'liblinear', 'max_iter': 200}`
  - `batch_size`: 64
  - `num_epochs`: 2
  - `body_learning_rate`: 1.5e-5
- Sets up StratifiedKFold with 5 splits.
- For each fold:
  - Trains SetFit model from scratch with fixed hyperparameters.
  - Evaluates on validation fold.
  - Logs classification report.

------

### 6️⃣ `06_model-outcome_predict.py`

**Purpose**: Apply the *trained binary model* to all study text. The predicted outcomes are then used for domain classification. 

- Uses the trained binary model to predict *"outcome" vs "not outcome"* for all manually extracted verbatim outcomes and negatives.
- Writes out CSV with predictions and training inclusion flags.

------

### 7️⃣ `07_model_outcome-domains_train-predict.py`

**Purpose**: Train the *multiclass outcome-domain classifier* to assign COMET outcome domain to verbatim outcome.

- Filters only *predicted outcomes* from previous step.
- Cleans labels (fixes naming inconsistencies).
- Excludes unwanted labels (resource use outcomes).
- Splits into train/eval/test sets.
- Uses same hyperparameters as in step 3 and 4 based on step 2 trials:
  - `head_params`: `{'solver': 'liblinear', 'max_iter': 200}`
  - `batch_size`: 64
  - `num_epochs`: 2
  - `body_learning_rate`: 1.5e-5
- Trains SetFit multiclass model to predict *Outcome Domains* using the [`thenlper/gte-base`](https://huggingface.co/thenlper/gte-base) model as the text encoder.
- Predicts labels for all verbatim outcomes identified in step 6..

------

### 8️⃣ `08_prep-output.py`

**Purpose**: Aggregate all evaluation metrics. This step prepares metrics for analysis, reporting, and visualization. 

- Reads JSON reports of binary and domain models.
- Extracts and flattens confusion matrices, precision, recall, F1-scores.
- Writes CSVs of metrics for binary and domain models.
- Creates confusion matrices for analysis.

------

## Unified PDF Parser

This repository also includes a *PDF parsing and outcome extraction module* (provided in `modules/__init__.py`), which can be run via `py-scripts/pdf_parse.py`.

It is designed to process *full-text PDF articles* and both extract *and* classify verbatim outcomes based on trained models:

### Parser steps

This parser provides an automated pipeline for:

1. **Parsing the text layer of a PDF**
   - Uses `pdfminer` to extract the entire text from academic PDFs.
   - Cleans and segments the document into sections based on headings.
2. **Identifying the Results Section**
   - Uses sentence embeddings (via `SentenceTransformer`) to identify section headings most semantically similar to “Results” and “Discussion.”
   - Extracts text strictly between these sections to focus on empirical results.
3. **Extracting Noun Phrases from Results**
   - Segments the Results section into sentences.
   - Applies spaCy NLP to extract *enhanced noun phrases*, including:
     - Base noun phrases
     - Associated adjectives
     - Prepositional phrases for context
   - Removes duplicates and overlapping phrases.
4. **Applying an Outcome Classification Model to Identify Verbatim Outcomes**
   - Loads a *pretrained binary outcome classifier* (SetFit model) to classify each extracted phrase as either:
     - `"outcome"`
     - `"not outcome"`
   - Returns only phrases labeled as *"outcome"*, with their classification probability.
5. **Assigning Each Outcome to a Predefined COMET Domain**
   - Uses a *pretrained multiclass SetFit model* to assign each verbatim outcome to one of the COMET framework’s outcome domains.
   - Adds this domain label to the final structured output.

------

### How It Works Internally

- **PDF Text Extraction**:
  - `pdfminer` reads raw text while preserving layout.
  - Regular expressions identify section breaks.
- **Results Section Identification**:
  - All headings are embedded with a SentenceTransformer.
  - Cosine similarity determines the best matches to “Results” and “Discussion.”
  - Ensures that "Results" precedes "Discussion" in document flow.
- **Noun Phrase Extraction**:
  - Uses `spaCy` for robust NLP parsing.
  - Includes premodifiers (adjectives) and prepositional phrases for richer phrases.
  - Prevents duplication by tracking token coverage.
- **Outcome Classification**:
  - Loads `SetFit` binary model (trained in Steps 4/5).
  - Classifies each noun phrase with the `predict_proba` method (using the wrapper function: `predict_withproba` to combine text, label and prediction probability).
  - Retains only those predicted as "outcome".
- **Outcome Domain Assignment**:
  - Loads `SetFit` multiclass model (from Step 7).
  - Assigns each "outcome" phrase to a COMET domain.

------

### Typical Usage

Once the binary and domain classification models are trained and saved, you can use this parser to *batch process new PDFs*:

```python
from parser import process_pdf

results = process_pdf("/path/to/article.pdf")

for outcome in results:
    print(outcome["text"], "->", outcome["outcome_domain"])
```

Returns a structured list of dictionaries with:

- Extracted phrase
- Sentence context
- Binary classification probability
- Assigned COMET domain
- Source filename

While the parser itself returns structured **in-memory results**, it can easily be adapted to:

- Write extracted outcomes to CSV.
- Store JSON for integration into systematic reviews.

The script `py-scripts/pdf_parse.py` contains code for applying the parser to a directory of pdfs.

### Integration with Pipeline

This module is fully compatible with:

- The binary outcome classification model trained in Steps 2–5.
- The outcome-domain classification model trained in Step 7.

Together, they enable fully automated extraction and classification of outcomes from PDFs for review work.

------


## Excluded Directories

The following directories are used in the scripts but excluded from the repository:

- **/data**: Raw and processed CSVs of outcomes and negatives.
- **/output**: All model outputs (predictions, metrics, confusion matrices).
- **/plots**: Any generated figures (e.g., performance vs training size).
- **/models**: Saved SetFit model artifacts for reuse.

------

## Dependencies

- Python 3.10+
- `setfit`
- `datasets`
- `optuna`
- `pandas`, `numpy`
- `scikit-learn`
- `plotnine`
- `pdfminer`
- `sentence-transformers`
- `pysbd`
- `spacy`



The `requirements_snapshot.txt` file contains a snapshot of the environment used for developing and running the pipeline.



## Other files

### `aicos_modelling_v2.qmd`

A Quarto markdown document for creating the technical report of the modelling and analysis (based on the outputs from step 8).



### `r-scripts/potting.r`

An R-script using `ggplot` to create the plot used in the article.



### `py-scripts/utils`

Various utility scripts for file conversion and interactive model testing.

Directory includes the scripts `final_binmodel_auroc_auprc.py` and `final_odmodel_auroc_auprc.py`, that calculates further evaluation metrics for trained SBERT models (like AUROC and AUPRC).

------

## License

MIT License. See `LICENSE` file for details.

------

## Acknowledgements

- Built using [spaCy](https://spacy.io/) for extracting candidate noun phrases and [SetFit](https://huggingface.co/docs/setfit/index) for few-shot learning using [`thenlper/gte-base`](https://huggingface.co/thenlper/gte-base) as text encoder.
