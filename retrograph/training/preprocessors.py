#####################################################
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
# Copyright 2019 Anne Lauscher, Nikolai Rozanov, Olga Majewska, Leonardo Ribeiro, Goran Glavas
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
####################################################


####################################################
# IMPORT STATEMENTS
####################################################
# >>>>>>  Native Imports  <<<<<<<
import os

# >>>>>>  Package Imports <<<<<<<
import tensorflow as tf
import csv

# >>>>>>  Local Imports   <<<<<<<
from retrograph.models import tokenization


####################################################
# CODE
####################################################
class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir, matched=True):
    """See base class."""
    if matched:
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
          "dev_matched")
    else:
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
          "dev_mismatched")


  def get_test_examples(self, data_dir, matched=True):
    """See base class."""
    if matched:
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")
    else:
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test")


  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class DiagnosticProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""


  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "diagnostic.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[1])
      text_b = tokenization.convert_to_unicode(line[2])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class WNLIProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")),
        "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[1])
      text_b = tokenization.convert_to_unicode(line[2])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


'''Added by Anne'''
class SST2Processor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[0])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class QQPProcessor(DataProcessor):
  """Processor for the QQP data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        text_b = tokenization.convert_to_unicode(line[2])
        label = "0"
      else:
        if len(line) != 6:
          # there is a problematic line
          print(line)
          continue
        text_a = tokenization.convert_to_unicode(line[3])
        text_b = tokenization.convert_to_unicode(line[4])
        label = tokenization.convert_to_unicode(line[5])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class QNLIProcessor(DataProcessor):
  """Processor for the QQP data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[1])
      text_b = tokenization.convert_to_unicode(line[2])
      if set_type == "test":
        label = "entailment"
      else:
        label = tokenization.convert_to_unicode(line[3])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class STSBProcessor(DataProcessor):
  """Processor for the STS-B data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      if set_type == 'test':
        text_a = tokenization.convert_to_unicode(line[-2])
        text_b = tokenization.convert_to_unicode(line[-1])
        label = 0.0
      else:
        text_a = tokenization.convert_to_unicode(line[-3])
        text_b = tokenization.convert_to_unicode(line[-2])
        label = float(line[-1])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class RTEProcessor(DataProcessor):
  """Processor for the RTE data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["entailment", "not_entailment"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[1])
      text_b = tokenization.convert_to_unicode(line[2])
      if set_type == "test":
        label = "entailment"
      else:
        label = tokenization.convert_to_unicode(line[3])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class SICKEntailmentProcessor(DataProcessor):
  """Processor for the SICK data set (SentEval version)."""

  def loadFile(self, fpath):
    skipFirstLine = True
    sick_data = {'X_A': [], 'X_B': [], 'y': []}
    with os.open(fpath, 'r', encoding='utf-8') as f:
      for line in f:
        if skipFirstLine:
          skipFirstLine = False
        else:
          text = line.strip().split('\t')
          sick_data['X_A'].append(text[1].split())
          sick_data['X_B'].append(text[2].split())
          sick_data['y'].append(text[4])
    return sick_data

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(self.loadFile(os.path.join(data_dir, 'SICK_train.txt')), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(self.loadFile(os.path.join(data_dir, 'SICK_trial.txt')), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(self.loadFile(os.path.join(data_dir, 'SICK_test_annotated.txt')), "test")

  def get_labels(self):
    """See base class."""
    return ['CONTRADICTION', 'NEUTRAL', 'ENTAILMENT']

  def _create_examples(self, dicts, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, dict) in enumerate(dicts):
      guid = "%s-%s" % (set_type, str(i))
      text_a = tokenization.convert_to_unicode(dict['X_A'])
      text_b = tokenization.convert_to_unicode(dict['X_B'])
      label = tokenization.convert_to_unicode(dict['y'])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class TRECProcessor(DataProcessor):
  """Processor for the TREC data set (SentEval version)."""

  def loadFile(self, fpath):
    trec_data = {'X': [], 'y': []}
    with os.open(fpath, 'r', encoding='latin-1') as f:
      for line in f:
        target, sample = line.strip().split(':', 1)
        sample = sample.split(' ', 1)[1].split()
        trec_data['X'].append(sample)
        trec_data['y'].append(target)
    return trec_data

  def get_train_examples(self, data_dir):
    """See base class."""
    data = self.loadFile(os.path.join(data_dir, 'train_5500.label'))
    split_index = len(data)*0.7
    return self._create_examples(data[:split_index], "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    data = self.loadFile(os.path.join(data_dir, 'train_5500.label'))
    split_index = len(data)*0.7
    return self._create_examples(data[split_index:], "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(self.loadFile(os.path.join(data_dir, 'TREC_10.label')), "test")

  def get_labels(self):
    """See base class."""
    return ['ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM']

  def _create_examples(self, dicts, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, dict) in enumerate(dicts):
      guid = "%s-%s" % (set_type, str(i))
      text_a = tokenization.convert_to_unicode(dict['X'])
      label = tokenization.convert_to_unicode(dict['y'])
      examples.append(
          InputExample(guid=guid, text_a=text_a, label=label))
    return examples


####################################################
# MAIN
####################################################


# EOF
