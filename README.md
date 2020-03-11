# Readme for Retrograph


## 0 - Downloading GLUE data and Pretrained BERT model
Step1: (download bert)
python3 download_bert.py

Step2: (downlaod glue)
python3 download_glue.py --data_dir data/GLUE --tasks all


## 1 - Creating Random Walks

Step1: (preprocess constraints)
preprocess_cn.py -> formats olga constraint into node-2-vec input
it has a relation_dict (which creates natural language) -> output is a tsv file (formated.tsv) "cn_assertions_filters.tsv"

Step2: (generate walks through graph)
random_walk.py -> takes the formated.tsv as input and creates a graph
it creates a random walk file. -> output("file_name.p") pickle file (it contains lists)

Step3: (generate corpus)
create_corpora_from_random_walks.py -> takes as input the pickle file and generates the corpus -> output corpus "rw_.txt" || could change how sentences are generated. at the moment sentences are always 3 word sentences -> if you want extra vocab in bert change function "create_realtionship_token"

Step4: (create pretraining out of corpus)
create_pretraining_data.py OR
create_pretraining_data_wo_nsp.py (without Next Sentence Prediciton)


## 1.1 - OMCS Pretraining Data

Step1: (create pretraining out of corpus)
create_pretraining_data.py OR
create_pretraining_data_wo_nsp.py (without Next Sentence Prediciton)

<!-- Comment -->
For OMSC you only need to create the pretraining data


## 2- Pretraining (Adapter)

Step1: (run the pretraining)
run_pretraining_adapter.py OR
run_pretraining_adapter_wo_nsp.py (without Next Sentence Prediciton)

<!-- Comment -->
Need to load the Adapter Model
And need to load the Adapter Optimiser for that.

## 3 - Finetuning (Adapter)

Step1:
run_classifier_adapter_tune_all.py ->

<!-- Comment -->
Need to load the Adapter Model
Here it is probably recommended to use the orginal optimiser as it optimises BERT


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
