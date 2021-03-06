#!/usr/bin/env bash

#Step1:
#run_classifier_adapter_tune_all.py ->
#
#<!-- Comment -->
#Need to load the Adapter Model
#Here it is probably recommended to use the orginal optimiser as it optimises BERT
TRAINING_UTILITY=training_utility

export CUDA_VISIBLE_DEVICES=0

BERT_DIR="models/BERT_BASE_UNCASED"
BERT_CONFIG=$BERT_DIR/bert_config.json
BERT_VOCAB=$BERT_DIR/vocab.txt

TASKNAME='COPA'
DATA_DIR=data/$TASKNAME

LEARNING_RATE=2e-5
EPOCHS=3.0
VARIANT=A

EXPERIMENT_NAME=$LEARNING_RATE.$EPOCHS$VARIANT

# BERT_EXTENDED_DIR="models/omcs_pretraining_free_wo_nsp_adapter"
# CHECKPOINT=${BERT_EXTENDED_DIR}/model.ckpt-${STEP}

BERT_EXTENDED_DIR=$BERT_DIR
CHECKPOINT=${BERT_EXTENDED_DIR}/bert_model.ckpt
OUTPUT_DIR="models/output_model_finetunning/${TASKNAME}/BERT_BASE/${EXPERIMENT_NAME}"


python3.6 $TRAINING_UTILITY/run_copa.py \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_VOCAB \
  --bert_config_file=$BERT_CONFIG \
  --init_checkpoint=$CHECKPOINT \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --learning_rate=$LEARNING_RATE \
  --num_train_epochs=$EPOCHS \
  --variant=$VARIANT \
  --output_dir=$OUTPUT_DIR/ | tee $OUTPUT_DIR.out
