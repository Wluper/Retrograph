#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=1
BERT_DIR="/home/Anne/uncased_L-12_H-768_A-12"
BERT_CONFIG=$BERT_DIR/bert_config.json
VOCAB_DIR=$BERT_DIR/vocab.txt
PATH_SUFFIX="/sentences/free-wo-nsp"
BERT_EXTENDED_DIR="/home/Anne/ConceptBERT/output/pretraining${PATH_SUFFIX}"
OUTPUT_DIR="/home/Anne/ConceptBERT/output/finetuning${PATH_SUFFIX}"
GLUE_DIR="/home/Anne/ConceptBERT/data/glue_data"
S3_PATH="~/anne/output/finetuning${PATH_SUFFIX}"

for STEP in "25000" "50000" "75000" "100000"; do
    CHECKPOINT=${BERT_EXTENDED_DIR}/model.ckpt-${STEP}
    for task_name in "CoLA" "MRPC" "RTE" "SST2" "QNLIV2" ; do
        echo $task_name
        echo $CHECKPOINT

        GLUE_DATA="$GLUE_DIR/$task_name"

        python run_classifier.py   \
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
        --output_dir=$OUTPUT_DIR/${STEP}/${task_name} |& tee $OUTPUT_DIR/${STEP}/${task_name}.out

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
        rm -r $OUTPUT_DIR/${STEP}/${task_name}*
    done



    for task_name in "STSB" ; do
        echo $task_name
        export GLUE_DATA="$GLUE_DIR/$task_name"

        python run_regression.py   \
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
        --output_dir=$OUTPUT_DIR/${STEP}/${task_name}  |& tee $OUTPUT_DIR/${STEP}/${task_name}.out

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
        rm -r $OUTPUT_DIR/${STEP}/${task_name}*
    done
done