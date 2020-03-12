#!/bin/bash

# Preprocess the relations
python preprocess_cn.py

mkdir -p 'randomwalks'

# Create the randomwalks using node2vec
python random_walks.py

