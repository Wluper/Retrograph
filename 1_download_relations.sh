#!/bin/bash

DOWNLOAD_UTILITY_SCRIPTS=download_utility


DIR_SAVE_RELATIONS='relations/'
mkdir -p $DIR_SAVE_RELATIONS

# DOWNLOAD RELATIONS
python3.6 $DOWNLOAD_UTILITY_SCRIPTS/download_relations.py --data_dir $DIR_SAVE_RELATIONS --relations all
