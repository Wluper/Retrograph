#!/usr/bin/env bash

#Step1:
#run_classifier_adapter_tune_all.py ->
#
#<!-- Comment -->
#Need to load the Adapter Model
#Here it is probably recommended to use the orginal optimiser as it optimises BERT


export CUDA_VISIBLE_DEVICES=8

BERT_DIR="models/BERT_BASE_UNCASED"
BERT_CONFIG=$BERT_DIR/bert_config.json
VOCAB_DIR=$BERT_DIR/vocab.txt

BERT_EXTENDED_DIR="data/output_pretrain_adapter"
OUTPUT_DIR="data/output_model_finetunning"

GLUE_DIR='data/GLUE'

OUTPUT_SUFFIX=_tune_all
### the second finetuning variant
for STEP in "98000" "99000"; do
    CHECKPOINT=${BERT_EXTENDED_DIR}/model.ckpt-${STEP}
    for task_name in "QNLI" "QQP" "MNLI"; do
        echo $task_name
        echo $CHECKPOINT

        GLUE_DATA="$GLUE_DIR/$task_name"

        python run_classifier_adapter_tune_all.py   \
        --task_name=$task_name \
        --do_train=true \
        --do_eval=true \
        --do_early_stopping=false \
        --data_dir=$GLUE_DATA \
        --vocab_file=$VOCAB_DIR \
        --bert_config_file=$BERT_CONFIG \
        --init_checkpoint=$CHECKPOINT\
        --max_seq_length=128 \
        --train_batch_size="[16]" \
        --learning_rate="[2e-5, 3e-5]" \
        --num_train_epochs="[3,4]" \
        --original_model=True \
        --output_dir=${OUTPUT_DIR}${OUTPUT_SUFFIX}/${STEP}/${task_name} |& tee ${OUTPUT_DIR}${OUTPUT_SUFFIX}/${STEP}/${task_name}.out
    done
done
