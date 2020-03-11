#!/bin/bash


DIR_SAVE_RELATIONS='relations/'

mkdir -p DIR_SAVE_RELATIONS

# DOWNLOAD RELATIONS
python download_relations.py --data_dir $DIR_SAVE_RELATIONS --relations all

mkdir -p 'data/GLUE'
mkdir -p 'models/BERT_BASE_UNCASED'

# DOWNLOAD BERT
python download_bert.py

# DOWNLOAD GLUE
python download_glue.py --data_dir data/GLUE --tasks all


