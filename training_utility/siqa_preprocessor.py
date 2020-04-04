# Nikolai Rozanov
from retrograph.modeling import tokenization
import tensorflow as tf
import os
import json
import numpy as np

class InputExample(object):
  """A single multiple choice question."""

  def __init__(
      self,
      qid,
      question,
      answers,
      label):
    """Construct an instance."""
    self.qid = qid
    self.question = question
    self.answers = answers
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
  def _read_json(cls, input_file):
    """Reads a JSON file."""
    with tf.gfile.Open(input_file, "r") as f:
      return json.load(f)

  @classmethod
  def _read_jsonl(cls, input_file):
    """Reads a JSON Lines file."""
    with tf.gfile.Open(input_file, "r") as f:
      return [json.loads(ln) for ln in f]


class SIQAProcessor(DataProcessor):
  """Processor for the CommonsenseQA data set."""

  LABELS = [0, 1, 2]

  TRAIN_FILE_NAME = 'socialIQa_v1.4_trn.jsonl'
  DEV_FILE_NAME = 'socialIQa_v1.4_dev.jsonl'
  TEST_FILE_NAME = 'socialIQa_v1.4_tst.jsonl'

  def __init__(self, variant="A"):
    """ There are four variants:
    Variant A: PREMISE [SEP] The cause/result was that ANSWER1 [SEP] The cause/result was that ANSWER2
    Variant B: PREMISE [SEP] What was the cause/result of ANSWER1 [SEP] What was the cause/result of ANSWER2
    Variant C: What was the cause/result of PREMISE [SEP] ANSWER1 [SEP]  ANSWER2
    Variant D: PREMISE [SEP] ANSWER1 [SEP] ANSWER2

    """
    self.variant = variant


  def get_train_examples(self, data_dir):
    train_file_name = self.TRAIN_FILE_NAME

    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, train_file_name)),
      'train')

  def get_dev_examples(self, data_dir):
    dev_file_name = self.DEV_FILE_NAME

    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, dev_file_name)),
      'dev')

  def get_test_examples(self, data_dir):
    test_file_name = self.TEST_FILE_NAME

    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, test_file_name)),
      'test')

  def get_labels(self):
    return [0, 1, 2]

  def _create_examples(self,lines, set_type):
    """ Calls one of the variants"""
    if self.variant=="A":
      return self._create_examples_variant_A(lines, set_type)
    elif self.variant=="B":
      return self._create_examples_variant_B(lines, set_type)
    elif self.variant=="C":
      return self._create_examples_variant_C(lines, set_type)
    elif self.variant=="D":
      return self._create_examples_variant_D(lines, set_type)
    else:
      raise Exception("NO SUCH VARIAN FOR COPA PREPROCESSING")


  ## VARIANT_A Premise [SEP] STATMENT_Answer [SEP] ST Answer
  def _create_examples_variant_A(self, lines, set_type):
    examples = []
    for line in lines:
      qid = line['idx']
      premise = tokenization.convert_to_unicode(line['premise'])

      question = line["question"]
      answers = np.array([
        tokenization.convert_to_unicode(question + line["choice1"]),
        tokenization.convert_to_unicode(question + line["choice2"]),
        tokenization.convert_to_unicode(question + line["choice3"])
        ])

      # the test set has no answer key so use '0' as a dummy label
      label = line.get('label', 0)

      examples.append(
        InputExample(
          qid=qid,
          question=premise,
          answers=answers,
          label=label))

    return examples

  ## VARIANT_B Premise [SEP] WH-Question_Answer [SEP] WH_Q Answer
  def _create_examples_variant_B(self, lines, set_type):
    examples = []
    for line in lines:
      qid = line['idx']
      question = line["question"]
      premise = tokenization.convert_to_unicode(line['premise'])

      answers = np.array([
        tokenization.convert_to_unicode(question + line["choice1"]),
        tokenization.convert_to_unicode(question + line["choice2"]),
        tokenization.convert_to_unicode(question + line["choice3"])
        ])

      # the test set has no answer key so use '0' as a dummy label
      label = line.get('label', 0)

      examples.append(
        InputExample(
          qid=qid,
          question=premise,
          answers=answers,
          label=label))

    return examples


  ## VARIANT_C WH-Question_Premise [SEP] Answer [SEP] Answer
  def _create_examples_variant_C(self, lines, set_type):
    examples = []
    for line in lines:
      qid = line['idx']
      question = line["question"]
      premise = tokenization.convert_to_unicode(question + line['premise'])

      answers = np.array([
        tokenization.convert_to_unicode(line["choice1"]),
        tokenization.convert_to_unicode(line["choice2"]),
        tokenization.convert_to_unicode(line["choice3"])
        ])

      # the test set has no answer key so use '0' as a dummy label
      label = line.get('label', 0)

      examples.append(
        InputExample(
          qid=qid,
          question=premise,
          answers=answers,
          label=label))

    return examples


  ## Premise WH-Question [SEP] Answer [SEP] Answer
  def _create_examples_variant_D(self, lines, set_type):
    examples = []
    for line in lines:
      qid = line['idx']
      question = line["question"]
      premise = tokenization.convert_to_unicode(line['premise'] + question)

      answers = np.array([
        tokenization.convert_to_unicode(line["choice1"]),
        tokenization.convert_to_unicode(line["choice2"]),
        tokenization.convert_to_unicode(line["choice3"])
        ])

      # the test set has no answer key so use '0' as a dummy label
      label = line.get('label', 0)

      examples.append(
        InputExample(
          qid=qid,
          question=premise,
          answers=answers,
          label=label))

    return examples
