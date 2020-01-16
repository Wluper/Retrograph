#!/usr/bin/env bash
# general config
export CUDA_VISIBLE_DEVICES=;
BERT_BASE_DIR="/work/anlausch/uncased_L-12_H-768_A-12"
VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
GLUE_DATA="$GLUE_DIR"
STEP_NUMBER=25000

# root dir of your checkpoints
# ROOT="/work/anlausch/ConceptBERT/output/finetuning/omcs/free-wo-nsp-adapter_tune_all/${STEP_NUMBER}/"
ROOT="/work/anlausch/ConceptBERT/output/finetuning/rw/1.0_1.0_2_10/nl-adapter_tune_all/${STEP_NUMBER}/"

# this is the mnli model which was best on the matched dataset
for config in "MNLI_16_2e-05_3/model.ckpt-73631","diagnostic"; do
    IFS=","
    set -- $config
    echo $1 and $2
    TASK=$2

    # location of the checkpoint which was best on dev
    TRAINED_CLASSIFIER=${ROOT}${1}
    OUTPUT_DIR=${ROOT}predictions/${TASK}
    # the actual prediction -- it is important to specify the checkpoint and to set train and eval to false but predict to true
    python run_classifier_adapter_tune_all.py \
      --task_name=${TASK} \
      --do_predict=true \
      --do_train=false \
      --do_eval=false \
      --data_dir=$GLUE_DIR/${TASK} \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$TRAINED_CLASSIFIER \
      --do_early_stopping=false \
      --max_seq_length=128 \
      --original_model=True \
      --matched=False \
      --output_dir=${OUTPUT_DIR}

    # this is a parser I wrote which should output the predictions in the glue platform format
    python parse_predictions.py \
    --task=${TASK} \
    --input_path="${OUTPUT_DIR}_32_5e-05_3.0/test_results.tsv" \
    --output_path_root="${OUTPUT_DIR}_32_5e-05_3.0"
done