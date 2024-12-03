#!/usr/bin/env bash

pip install --upgrade pip
pip install -r /work/aicos/requirements.txt

python -m spacy download 'en_core_web_sm'