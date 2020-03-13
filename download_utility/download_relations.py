"""
This is the script for downloading ConceptNet relations from S3.
Make sure waws is installed and configured:

pip3 install waws
waws --configure

To run: python3 download_relations.py
To download all (31): -r all (default)
To download specific relations, provide a comma-separated list of relations: -r isA,formOf
To specify/create a local download directory, use: -d directory_name.
"""

import os
import sys
import argparse
import waws

RELATIONS = ['relatedTo', 'formOf', 'isA', 'partOf', 'hasA', 'usedFor', 'capableOf', 
            'atLocation', 'causes', 'hasSubevent', 'hasFirstSubevent', 'hasLastSubevent', 
            'hasPrerequisite', 'hasProperty', 'motivatedByGoal', 'obstructedBy', 'desires', 
            'createdBy', 'synonyms', 'antonyms', 'distinctFrom', 'derivedFrom', 'symbolOf', 
            'definedAs', 'mannerOf', 'locatedNear', 'hasContext', 'similarTo', 'causesDesire', 
            'madeOf', 'receivesAction']

s3 = waws.BucketManager()

def download(relation, data_dir):
    print("Downloading and extracting %s..." % relation)
    data_file = "cn_%s.txt" % relation
    
    s3.download_file(
    file_name=data_file,
    local_path=data_dir,
    remote_path="",
    bucket_name="wluper-retrograph"
    )
    
    print("\tDone!")

def get_relations(relation_names):
    relation_names = relation_names.split(',')
    if "all" in relation_names:
        relations = RELATIONS
    else:
        relations = []
        for rel_name in relation_names:
            assert rel_name in RELATIONS, "Relation %s not found!" % rel_name
            relations.append(rel_name)
    return relations

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', help='directory to save data to', type=str, default='./')
    parser.add_argument('-r', '--relations', help='relations to download as a comma separated string',
                        type=str, default='all')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    relations = get_relations(args.relations)

    for rel in relations:
        download(rel, args.data_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
