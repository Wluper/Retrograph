#!/usr/bin/env bash
#--input_file=./data/omcs-sentences-more-filtered.txt \
#--output_file=./data/omcs-sentences-more-filtered.tfrecord \

python create_pretraining_data.py \
--input_file=./data/omcs-sentences-free-filtered-3.txt \
--output_file=./data/omcs-sentences-free-filtered.tfrecord \
--vocab_file=/c/Users/anlausch/Downloads/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/vocab.txt \
--do_lower_case=True \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=5
