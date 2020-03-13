''' Script for downloading all CommonsenseQA data.
Author: Nikolai Rozanov
'''

import os
import sys
import shutil
import argparse
import tempfile
import urllib.request


LINKS = [
  "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl",
  "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl",
  "https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl"
]

def download_and_extract(link, data_dir):
  """ downloads and moves. """
  print("Downloading and extracting %s..." % link)
  data_file = get_name_from_link(link)
  urllib.request.urlretrieve(link,data_file)
  shutil.move(data_file,os.path.join(data_dir,data_file))
  print("\tCompleted!")

def get_name_from_link(link):
  """ returns name from link. """
  name = link.split("/")[-1]
  return name

def make_dir(directory_path, directory_name):
  """ Makes a directory if it doesn't exist. """
  directory = os.path.join(directory_path, directory_name)
  if not os.path.exists(directory):
      os.makedirs(directory)

def main():
  DATA="data"
  TARGET_FOLDER="COMMONSENSEQA"
  data_dir = os.path.join(DATA, TARGET_FOLDER)

  make_dir(DATA, TARGET_FOLDER)
  for link in LINKS:
    download_and_extract(link, data_dir)



if __name__ == '__main__':
  main()
