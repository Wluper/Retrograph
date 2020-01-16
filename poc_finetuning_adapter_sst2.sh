#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1
BERT_DIR="/work/anlausch/uncased_L-12_H-768_A-12"
BERT_CONFIG=$BERT_DIR/bert_config.json
VOCAB_DIR=$BERT_DIR/vocab.txt


PATH_SUFFIX="/rw/1.0_1.0_2_10/nl-adapter"
OUTPUT_SUFFIX=_tune_all_quick_insight
BERT_EXTENDED_DIR="/work/anlausch/ConceptBERT/output/pretraining${PATH_SUFFIX}"
OUTPUT_DIR="/work/anlausch/ConceptBERT/output/finetuning${PATH_SUFFIX}"

### the second finetuning variant
for STEP in "25000" "100000"; do
    CHECKPOINT=${BERT_EXTENDED_DIR}/model.ckpt-${STEP}
    for task_name in "SST2"; do
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
