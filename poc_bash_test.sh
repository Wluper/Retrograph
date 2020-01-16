#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=1
export BERT_DIR="/home/Anne/uncased_L-12_H-768_A-12"
export BERT_CONFIG=$BERT_DIR/bert_config.json
export VOCAB_DIR=$BERT_DIR/vocab.txt
export PATH_SUFFIX="/sentences/free-wo-nsp"
export BERT_EXTENDED_DIR="/home/Anne/ConceptBERT/output/pretraining${PATH_SUFFIX}"
export OUTPUT_DIR="/home/Anne/ConceptBERT/output/finetuning${PATH_SUFFIX}"
export GLUE_DIR="/home/Anne/ConceptBERT/data/glue_data"
export S3_PATH="~/test/output/finetuning${PATH_SUFFIX}"

for STEP in "25000"; do
    CHECKPOINT=${BERT_EXTENDED_DIR}/model.ckpt-${STEP}
    for task_name in "SST2"; do

        # Copy the data to s3
        for dir in ${OUTPUT_DIR}/${STEP}/*/; do
        #for dir in /home/Anne/ConceptBERT/output/finetuning/sentences/free-wo-nsp/25000/CoLA*; do
            echo "DIR ${dir}"
            for filename in ${dir}*; do
                echo "FILENAME ${filename}"

                #IFS='/' # hyphen (-) is set as delimiter
                #declare -a PARTS
                #read -ra PARTS <<< ${FILE} # str is read into an array as tokens separated by IFS
                #echo "PARTS ${PARTS}"
                FILE=${filename##*/}
                echo ${FILE}
                temp=${filename%/*}
                SUBDIR=${temp##*/}
                echo ${SUBDIR}

                #S3="${S3_PATH}/${STEP}/${PARTS[${#PARTS[@]}-2]}/${PARTS[${#PARTS[@]}-1]}"
                S3=${S3_PATH}/${STEP}/${SUBDIR}/${FILE}
                #S3="${S3_PATH}/${STEP}/${filename}"
                echo "S3 ${S3}"
                waws --uploadS3 -b wluper-retrograph -f "${filename}" -l "${S3}"
            done
        done
        #waws --uploadS3 -b wluper-retrograph -f $OUTPUT_DIR/${STEP}/${task_name}/ -l $S3_PATH/${STEP}/${task_name}/
        #rm -r $OUTPUT_DIR/${STEP}/${task_name}*
    done
done