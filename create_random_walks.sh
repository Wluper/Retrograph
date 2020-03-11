#!/bin/bash

python preprocess_cn.py

mkdir -p 'randomwalks'

python random_walks.py

