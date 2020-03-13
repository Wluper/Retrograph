import codecs

"""
As we got the relations to consider from olga, we don't need to do this anymore
"""
# def filter_assertions(path="./relations/assertions.csv"):
#   assertions = []
#   with codecs.open(path, "r", "utf8") as f:
#     reader = csv.DictReader(f, dialect=csv.excel_tab, fieldnames=["URI", "relation", "node_a", "node_b", "info"])
#     for i,row in enumerate(reader):
#       node_a = row["node_a"].split("/c/en/")
#       node_b = row["node_b"].split("/c/en/")
#       if len(node_a) > 1 and len(node_b) > 1:
#         # these should be nodes in english
#         node_a = node_a[1].split("/")[-1].replace("_", "-")
#         node_b = node_b[1].split("/")[-1].replace("_", "-")
#         print(node_a)
#         print(node_b)

"""
Based on the relations from olga
"""
def create_joined_assertions_for_random_walks(paths=["./relations/cn_antonyms.txt", "./relations/cn_isA.txt", "./relations/cn_mannerOf.txt","./relations/cn_synonyms.txt"], output_path="./randomwalks/cn_assertions_filtered.tsv"):
  # we ideally want to have a "natural language representation" of the relations
  # TODO: keep in mind that antonymy and synonymy are bidirectional relationships, so maybe we want to account for this, i.e., by creating the corresponding pairs in the opposite direction or so
  # TODO: As an alternative of random walks, we can also just use the natural language representation of the relationships
  relation_dict = {
    "antonyms": "is an antonym of",
    "isA": "is a",
    "mannerOf": "is a manner of",
    "synonyms": "is a synonym of"
  }
  all_assertions = []
  for path in paths:
    relation = path.split("cn_")[1].split(".txt")[0]
    nl_relation = relation_dict[relation]
    with codecs.open(path, "r", "utf8") as f:
      for line in f.readlines():
        word_a, word_b = line.strip().split("\t")
        full_assertion = [word_a, word_b, nl_relation]
        all_assertions.append(full_assertion)
        # TODO: here is an attempt to account for bidirectionality; Does it make sense?
        if relation == "antonyms" or relation == "synonyms":
          full_assertion_b = [word_b, word_a, nl_relation]
          all_assertions.append(full_assertion_b)
  # In total, we have 293105 assertions
  print("In total, we have %d assertions" % len(all_assertions))
  with codecs.open(output_path, "w", "utf8") as out:
    for assertion in all_assertions:
      out.write(assertion[0] + "\t" + assertion[1] + "\t" + assertion[2] + "\n")



def main():
  create_joined_assertions_for_random_walks()
  #profile_data()
  #filter_assertions()

if __name__ == "__main__":
  main()
