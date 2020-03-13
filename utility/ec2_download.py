import waws

inst = waws.InstanceManager()

inst.download_from_EC2(folder_file_name="~/ConceptBERT/output/pretraining/sentences/free-wo-nsp", local_path="/c/Users/anlausch/Downloads/omcs", instance="sunshine-1")
