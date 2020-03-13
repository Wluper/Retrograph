# Readme for Retrograph

Environment: python 3.6

Please, follow these instructions to execute the experiments.

## 0 - Dependencies
```
pip install networkx
pip install tqdm
pip install waws
TODO add all dependencies
```

## 1 - Downloading project data
Step 1: GLUE data, Pretrained BERT model and relations
```
bash ./1_download_data_project.sh
```
It creates:
1. relations/cn_relationType*.txt
2. data/GLUE/TASK*
3. models/BERT_BASE_UNCASED

## 2 - Creating Random Walks

Step 2: Create the sequences of tokens using random walks generated by node2vec:
```
bash ./2_create_random_walks.sh
```

It creates the main file `randomwalks/random_walk_1.0_1.0_2_15.p` and others also (`randomwalks/cn_assertions_filtered.tsv`)



## 3 - Generating the Corpus (This takes a serious while)
Step 3: Create natural language text from the random walks:
```
bash ./3_generate_corpus.sh
```
The generated corpus will be used as input for BERT + Adapters. It creates a file in TF format: `randomwalks/rw_corpus_1.0_1.0_2_15_nl.tf` (and also generates: `randomwalks/rw_corpus_1.0_1.0_2_15_nl.tf`)


## 4 - Pretraining Adapter

Step 4: Pretrain the adapter using the RW corpus:
```
bash ./4_pretrain_adapter.sh
```
Creates a model in: `models/output_pretrain_adapter`


## 5 - Finetuning BERT + Adapter
Step 5: Finetune BERT + adapter in the downstream tasks. To execute a grid search for the hyperparameters, execute the following command:
```
bash ./5_finetune_adapter.sh
```
Creates a model in: `models/output_model_finetunning`


## 4 - GLUE Submission

Step1: (find best results from grid search)
fetcher.py -> helps you find the best model from a grid generated version of the model

Step2:
predictions_....sh

parse_prediction.py -> helps you create the right output file format, and you need to name the glue.zip submission folder in the right way as well.


## TODO:
1. CommonsenseQA:
- Preprocessor for CommonsenseQA
- Download CommonsenseQA dataset
- Evaluation scripts or standards?

github/jonathanherzig



<!-- EOF -->
