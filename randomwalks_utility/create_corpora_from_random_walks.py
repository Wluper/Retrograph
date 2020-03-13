import pickle
import codecs
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_walks(path="./randomwalks/random_walk_1.0_1.0_2_10.p"):
  return pickle.load(open(path, "rb"))


def create_relationship_token(text):
  # here, I've changed something
  return text #"<" + "".join(text.split(" ")) + ">"

def process_walks(walks):
  text = ""
  for walk in walks:
    previous_token = ""
    for i, token in enumerate(walk):
      # every first token is a node and every second is a relationship
      # we don't need to capitalize anything as we are anyways working with the uncased BERT
      if (i % 2 == 0 and previous_token != "" and i != 0 and i != 2) or (i == 3 and previous_token != ""):
        # we have reached the end of a valid sentence sequence, so we put a period
        if i == 3:
          text = text[:-1] + ".\n"
        else:
          text = text + token + ".\n"
        if i != len(walk) - 1 and i == 3:
          # if the walk is not finished yet, we duplicate the token
          text = text + previous_token + " " + create_relationship_token(token) + " "
        elif i != len(walk) - 1:
          # if the walk is not finished yet, we duplicate the token
          text = text + token + " "
        else:
          # otherwise we can put a new line to mark the end of a document
          text = text + "\n\n"
      elif i % 2 == 0:
        text = text + token + " "
      elif i % 1 == 0:
        text = text + create_relationship_token(token) + " "
      previous_token = token

  return text

def chunks(lst, n):
  """Yield successive n-sized chunks from lst."""
  for i in range(0, len(lst), n):
    yield lst[i:i + n]

def generate_corpus_from_walks(walks, output_path_prefix="./randomwalks/rw_corpus_", output_path_suffix=""):
  # how do we actually want to generate the corpus?
  # one option is to always dublicate the node in the middle..
  # also Goran says that we want to keep the relations as separate tokens in the vocab. I do not necessarily agree with this, but we try.
  # What is one document? Is is always one walk? Maybe yes...
  output_path = output_path_prefix + output_path_suffix + ".txt"
  text = ""
  print('size of walks', len(walks))
  print('processing RWs...')

  workers = 10
  splits = 1000
  text = ""
  with ProcessPoolExecutor(max_workers=workers) as executor:
    futures = {}
    for i, ws in enumerate(chunks(walks, splits)):
        job = executor.submit(process_walks, ws)
        futures[job] = i

    for job in tqdm(as_completed(futures)):
        t = job.result()
        text += t
        r = futures[job]
        del futures[job]


  with codecs.open(output_path, "w", "utf8") as out:
    out.write(text)


def main():
  in_prefix = "randomwalks/random_walk_"
  in_suffix = "1.0_1.0_2_15"
  walks = load_walks(in_prefix + in_suffix + ".p")
  generate_corpus_from_walks(walks, output_path_suffix=in_suffix + "_nl")


if __name__=="__main__":
  main()
