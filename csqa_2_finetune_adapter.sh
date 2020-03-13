#!/usr/bin/env bash

#Step1:
#run_classifier_adapter_tune_all.py ->
#
#<!-- Comment -->
#Need to load the Adapter Model
#Here it is probably recommended to use the orginal optimiser as it optimises BERT
TRAINING_UTILITY=training_utility

export CUDA_VISIBLE_DEVICES=8

BERT_DIR="models/BERT_BASE_UNCASED"
BERT_CONFIG=$BERT_DIR/bert_config.json
VOCAB_DIR=$BERT_DIR/vocab.txt

BERT_EXTENDED_DIR="models/output_pretrain_adapter"
OUTPUT_DIR="models/output_model_finetunning"
OUTPUT_SUFFIX=_tune_all

TASKNAME='COMMONSENSEQA'
DATA_DIR=data/$TASKNAME

python3.6 $TRAINING_UTILITY/run_commonsenseqa.py
  --split=$SPLIT \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR/
