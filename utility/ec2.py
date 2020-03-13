import waws

inst = waws.InstanceManager()

#inst.upload_to_EC2(folder_file_name=".", instance="sunshine-1")
#inst.upload_to_EC2(folder_file_name="./modeling.py",  optional_remote_path="./ConceptBERT/", instance="sunshine-1")
#inst.upload_to_EC2(folder_file_name="./data/glue_data/",  optional_remote_path="./ConceptBERT/data/", instance="sunshine-1")
inst.upload_to_EC2(folder_file_name="./poc_finetuning.sh",  optional_remote_path="./ConceptBERT/", instance="sunshine-1")
inst.upload_to_EC2(folder_file_name="./run_regression.py",  optional_remote_path="./ConceptBERT/", instance="sunshine-1")
inst.upload_to_EC2(folder_file_name="./run_classifier.py",  optional_remote_path="./ConceptBERT/", instance="sunshine-1")
inst.upload_to_EC2(folder_file_name="./poc_bash_test.sh",  optional_remote_path="./ConceptBERT/", instance="sunshine-1")
#inst.upload_to_EC2(folder_file_name="/c/Users/anlausch/Downloads/uncased_L-12_H-768_A-12/", instance="sunshine-1")

#inst.download_from_EC2(folder_file_name="CODE_FOLDER", local_path="./training", optional_remote_path="EXPERIMENT2", instance="sunshine-1")