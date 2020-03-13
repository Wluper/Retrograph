#!/bin/bash

#Step1: (run the pretraining)
#run_pretraining_adapter.py OR
#run_pretraining_adapter_wo_nsp.py (without Next Sentence Prediciton)
#
#<!-- Comment -->
#Need to load the Adapter Model
#And need to load the Adapter Optimiser for that.

export CUDA_VISIBLE_DEVICES=8

BERT_CONFIG_FILE=models/BERT_BASE_UNCASED/bert_config.json
INPUT_FILE=randomwalks/rw_corpus_1.0_1.0_2_15_nl.tf
OUTPUT_DIR=data/output_pretrain_adapter

mkdir -p $OUTPUT_DIR

python run_pretraining_wo_nsp_adapter.py --input_file $INPUT_FILE --output_dir $OUTPUT_DIR \
    --bert_config_file $BERT_CONFIG_FILE \
    --do_train True




