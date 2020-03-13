#!/bin/bash

RANDOM_WALKS_SCRIPTS=randomwalks_utility

# Preprocess the relations
python3.6 $RANDOM_WALKS_SCRIPTS/preprocess_cn.py

mkdir -p 'randomwalks'

# Create the randomwalks using node2vec
python3.6 $RANDOM_WALKS_SCRIPTS/random_walks.py
