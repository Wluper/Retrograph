#!/usr/bin/env bash

python create_pretraining_data.py \
--input_file=./data/rw_corpus_1.0_1.0_2_10_2.txt \
--output_file=./data/rw_corpus_1.0_1.0_2_10_cn_relations_2.tfrecord \
--vocab_file=/work/anlausch/uncased_L-12_H-768_A-12/vocab_cn_relations.txt \
--do_lower_case=True \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=5 |& tee ./data/cn_relations_2.out

python create_pretraining_data.py \
--input_file=./data/rw_corpus_1.0_1.0_2_10_3.txt \
--output_file=./data/rw_corpus_1.0_1.0_2_10_cn_relations_3.tfrecord \
--vocab_file=/work/anlausch/uncased_L-12_H-768_A-12/vocab_cn_relations_2.txt \
--do_lower_case=True \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=5 |& tee ./data/cn_relations_3.out

python create_pretraining_data.py \
--input_file=./data/rw_corpus_1.0_1.0_2_10_nl.txt \
--output_file=./data/rw_corpus_1.0_1.0_2_10_cn_relations_nl.tfrecord \
--vocab_file=/work/anlausch/uncased_L-12_H-768_A-12/vocab.txt \
--do_lower_case=True \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=5 |& tee ./data/cn_relations_nl.out
