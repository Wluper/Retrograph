#!/bin/bash

# Create natural language text from the RWs
# create_corpora_from_random_walks.py -> takes as input the pickle file and generates the corpus
# -> output corpus "rw_.txt"
# || could change how sentences are generated. at the moment sentences are always 3 word sentences
# -> if you want extra vocab in bert change function "create_realtionship_token"

RANDOM_WALKS_SCRIPTS=randomwalks_utility
DATA_SCRIPTS=data_utility

python3.6 $RANDOM_WALKS_SCRIPTS/create_corpora_from_random_walks.py

# COMMENTS - NIKOLAI
#create_pretraining_data.py OR
#create_pretraining_data_wo_nsp.py (without Next Sentence Prediciton)
#<!-- Comment -->
#For OMSC you only need to create the pretraining data
## 4 - Pretraining BERT using RW Corpus

## 1.1 - OMCS Pretraining Data
#Step1: (create pretraining out of corpus)
#create_pretraining_data.py OR
#create_pretraining_data_wo_nsp.py (without Next Sentence Prediciton)

VOCAB_FILE=models/BERT_BASE_UNCASED/vocab.txt
INPUT_FILE=randomwalks/rw_corpus_1.0_1.0_2_15_nl.txt
OUTPUT_FILE=randomwalks/rw_corpus_1.0_1.0_2_15_nl.tf


python3.6 $DATA_SCRIPTS/create_pretraining_data_wo_nsp.py --input_file $INPUT_FILE --output_file $OUTPUT_FILE --vocab_file $VOCAB_FILE
