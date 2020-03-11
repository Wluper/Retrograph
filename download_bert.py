# coding=utf-8
# Copyright 2019 Wluper Ltd. Team, Nikolai Rozanov.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# ##############################################################################
# Import
##############################################################################

# Native
import urllib.request
import os
import zipfile

# Packages
import shutil

# Local


# #############################################################################
# Code
##############################################################################
BERT_TO_URL_MAPPING = {
  "BERT_LARGE_UNCASED_WHOLEWORD"  : "https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip",
  "BERT_LARGE_CASED_WHOLEWORD"    : "https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip",

  "BERT_LARGE_UNCASED"            : "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip",
  "BERT_LARGE_CASED"              : "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip",

  "BERT_BASE_UNCASED"             : "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
  "BERT_BASE_CASED"               : "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip",

  "BERT_BASE_CASED_MULTI"         : "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip",#re
  "BERT_BASE_UNCASED_MULTI"       : "https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip",

  "BERT_BASE_CHINESE"             : "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip"
}


def download_bert_zip(target_file_name:str, which_bert:str="BERT_BASE_CASED"):
  """
  Downloads the officially pre-trained model from google.
  File is a zip and contains:
    1. A TensorFlow checkpoint (bert_model.ckpt) containing the pre-trained weights (which is actually 3 files).
    2. A vocab file (vocab.txt) to map WordPiece to word id.
    3. A config file (bert_config.json) which specifies the hyperparameters of the model.

  Part Reference:
  https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
  """
  try:
    url = BERT_TO_URL_MAPPING[which_bert]
  except KeyError:
    print("Seems like this BERT model doesn't exist. Please specify a possible option.")
    exit()
  os.makedirs(os.path.dirname(target_file_name),exist_ok=True) #creates path if not in existence.
  with urllib.request.urlopen(url) as response, open(target_file_name, 'wb') as out_file:
    print(f"Downloading: {which_bert}. Target_file: {target_file_name}\nThis may take some time.")
    shutil.copyfileobj(response, out_file)
  print("Finished the Download.")


def unzip_bert(path_to_zip_file: str, target_folder_name: str):
  """
  unzips the bert and places the content into target_folder_name.

  Part Reference:
  https://stackoverflow.com/questions/3451111/unzipping-files-in-python
  """
  print(f"Unzipping Bert zip {path_to_zip_file}.")
  with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
      zip_ref.extractall(target_folder_name)
  print("Finished Unzipping.")
  print(f"Moving Content to: {target_folder_name}")
  _move_unzipped_content(target_folder_name)
  print("Finished Moving. Finished Process.")


def _move_unzipped_content(target_folder_name:str):
  """ Helper function to move content for function unzip_bert. (has assumptions)"""
  bert_data = os.listdir(target_folder_name)[0]
  final_bert_data_path = os.path.join(target_folder_name, bert_data)
  for file in os.listdir(final_bert_data_path):
    shutil.move(os.path.join(final_bert_data_path,file), target_folder_name)
  os.rmdir(final_bert_data_path)

# #############################################################################
# MAIN
##############################################################################
if __name__=="__main__":
  which_bert = "BERT_BASE_UNCASED"
  target_file_name = os.path.join("models","bert_pretrained.zip")
  target_folder_name = os.path.join("models",which_bert)
  download_bert_zip(target_file_name=target_file_name, which_bert=which_bert)
  unzip_bert(target_file_name, target_folder_name)
