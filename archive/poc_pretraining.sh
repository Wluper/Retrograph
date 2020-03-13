#!/usr/bin/env bash
echo "script started"

INPUT_FILE="/home/Anne/ConceptBERT/data/omcs-sentences-free-filtered-wo-nsp.tfrecord"
OUTPUT_DIR="/home/Anne/ConceptBERT/output/pretraining/sentences/free-wo-nsp/"
NUM_TRAIN_STEPS=100000
BERT_DIR="/home/Anne/uncased_L-12_H-768_A-12"
BERT_CONFIG=$BERT_DIR/bert_config.json
# TODO: Here is an error!!! We should run this again and change run_pretraining to run_pretraining_wo_nsp
#
python run_pretraining.py \
--input_file=$INPUT_FILE \
--output_dir=$OUTPUT_DIR \
--do_train=True \
--do_eval=True \
--bert_config_file=$BERT_CONFIG \
--train_batch_size=16 \
--eval_batch_size=8 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=$NUM_TRAIN_STEPS \
--num_warmup_steps=1000 \
--learning_rate=1e-4 \
--max_eval_steps=1000 \
--save_checkpoints_steps=25000 \
--init_checkpoint=$BERT_DIR/bert_model.ckpt