#!/usr/bin/env bash
# general config
export CUDA_VISIBLE_DEVICES=0;
BERT_BASE_DIR="/work/anlausch/uncased_L-12_H-768_A-12"
VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
GLUE_DATA="$GLUE_DIR"
STEP_NUMBER=100000

ROOT="/work/anlausch/ConceptBERT/output/finetuning/rw/1.0_1.0_2_10/nl-adapter_tune_all/${STEP_NUMBER}/"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16/1000000/MNLI_16_3e-05_3/model.ckpt-73631"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16/1000000/MNLI_16_2e-05_3/model.ckpt-73631"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16/1000000/MNLI_16_2e-05_4/model.ckpt-98175"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16/1000000/MNLI_16_3e-05_4/model.ckpt-98175"

for config in "_16_3e-05_3","model.ckpt-73631" "_16_2e-05_3","model.ckpt-73631" "_16_2e-05_4","model.ckpt-98175" "_16_3e-05_4","model.ckpt-98175"; do
    IFS=","
    set -- $config
    echo $1 and $2
    SUFFIX=$1
    MODEL=$2

    TRAINED_CLASSIFIER="${ROOT}MNLI${SUFFIX}/${MODEL}"

    TASK="MNLI"
    GLUE_DATA="$GLUE_DIR/${TASK}"

    OUTPUT_DIR="${ROOT}MNLI-mm${SUFFIX}"

    python run_classifier_adapter_tune_all.py   \
    --task_name=$TASK \
    --do_train=False \
    --do_eval=true \
    --do_early_stopping=false \
    --data_dir=$GLUE_DATA \
    --vocab_file=$VOCAB_DIR \
    --bert_config_file=$BERT_CONFIG \
    --init_checkpoint=$TRAINED_CLASSIFIER \
    --max_seq_length=128 \
    --original_model=True \
    --matched=False \
    --output_dir=$OUTPUT_DIR/${STEP}/${TASK}-mm |& tee $OUTPUT_DIR/${STEP}/${TASK}-mm.out
done
