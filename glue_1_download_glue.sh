#!/bin/bash

DOWNLOAD_UTILITY_SCRIPTS=download_utility


DIR_SAVE_RELATIONS='relations/'
mkdir -p $DIR_SAVE_RELATIONS

# DOWNLOAD RELATIONS
python3.6 $DOWNLOAD_UTILITY_SCRIPTS/download_relations.py --data_dir $DIR_SAVE_RELATIONS --relations all

mkdir -p 'data/GLUE'
mkdir -p 'models/BERT_BASE_UNCASED'

# DOWNLOAD BERT
python3.6 $DOWNLOAD_UTILITY_SCRIPTS/download_bert.py

# DOWNLOAD GLUE
python3.6 $DOWNLOAD_UTILITY_SCRIPTS/download_glue.py --data_dir data/GLUE --tasks all
