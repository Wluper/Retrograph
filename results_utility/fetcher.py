import os
import codecs
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def fetch_results(base_path="/work/anlausch/ConceptBERT/output/finetuning/omcs/", subdir="free-wo-nsp/"):#subdir="base_16_longer/"):# ##
  """
  :param base_path:
  :param subdir:
  :return:
   >>> fetch_results()
  """
  path = base_path + subdir
  result_dict = {}
  for root, dirs, files in os.walk(path):
    for f in files:
      if f == "eval_results.txt":
        id = root.split("/")[-1]
        task = "_".join(id.split("_")[:1])
        hyperparams = "_".join(id.split("_")[1:])
        train_step = root.split("/")[-2]
        if train_step in result_dict:
          train_step_dict = result_dict[train_step]
        else:
          train_step_dict = {}
        with codecs.open(os.path.join(root, f), "r", "utf8") as file:
          file_dict = {}
          for line in file.readlines():
            key = line.split(" = ")[0]
            try:
              value = float(line.split(" = ")[1].strip())
              file_dict[key] = value
            except Exception as e:
              print(e)
          if task not in train_step_dict:
            train_step_dict[task] = {}
          train_step_dict[task][hyperparams] = file_dict
        result_dict[train_step] = train_step_dict
  filtered_list = []
  for train_step, train_step_dict in result_dict.items():
    for task, task_dict in train_step_dict.items():
      if task in ["MRPC", "RTE", "MNLI", "QQP", "SST2", "QNLI", "QNLIV2"]:
        measure = "eval_accuracy"
      elif task in ["CoLA"]:
        measure = "mcc"
      elif task in ["STSB"]:
        measure = "spearman"
      else:
        print("Task name not in list: %s", task)
      if len(task_dict) < 4:
        print("Task %s result dict for train step %s has not all hyperparam results" % (task, train_step))
        break
      else:
        best_config = ""
        best_result = 0.0
        for i, (config, result) in enumerate(task_dict.items()):
          if result[measure] >= best_result:
            best_config = config
            best_result = result[measure]
        filtered_list.append({"train_step": train_step, "task": task, "hyperparams": best_config, "score": best_result})
  return filtered_list


def output_results_as_csv(filtered_list, output_path="./../finetuning/poc_over_time/wn_binary.csv"):
  csv_keys = list(filtered_list[0].keys())
  with open(output_path, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, csv_keys)
    dict_writer.writeheader()
    dict_writer.writerows(filtered_list)

def plot_task(task="CoLA", output_path="./../finetuning/poc_over_time/cola2.pdf"):
  """
  :param output_path:
  :param task:
  :return:
   >>> plot_task()
  """
  filtered_list_wn = fetch_results(subdir="wn_binary") + fetch_results(subdir="wn_binary_16_longer")
  # kick out ill stsb
  for d in filtered_list_wn:
    if d["train_step"] != "stsb_first":
      d["train_step"] = int(d["train_step"])/2
      d["model"] = "informed"
  filtered_list_base = fetch_results(subdir="base_16") + fetch_results(subdir="base_16_longer")
  for d in filtered_list_base:
      d["train_step"] = int(d["train_step"])
      d["model"] = "base"
  filtered_list_wn = [d for d in filtered_list_wn if d["train_step"] != "stsb_first" and d["task"]==task and d["train_step"] == 1000000]
  filtered_list_base = [d for d in filtered_list_base if d["task"] == task and d["train_step"] == 1000000]
  filtered_list_wn = sorted(filtered_list_wn, key=lambda k: k['train_step'])
  filtered_list_base = sorted(filtered_list_base, key=lambda k: k['train_step'])

  # aligned_wn = []
  # aligned_base = []
  # ind = []
  # for d_base in filtered_list_base:
  #   for d_wn in filtered_list_wn:
  #     if int(d_base["train_step"])*2 == int(d_wn["train_step"]):
  #       aligned_base.append(d_base["score"])
  #       aligned_wn.append(d_wn["score"])
  #       ind.append(d_base["train_step"])
  #       break
  all = filtered_list_wn + filtered_list_base
  df = pd.DataFrame(all)

  sns.set()

  with sns.plotting_context("paper"):
    #ind = lm_steps  # the x locations for the groups

    fig, ax = plt.subplots()

    sns.lineplot(x="train_step", y="score", hue="model", style="model", data=df)
    #plt.title(task)

    ax.set(xlabel='Language Modeling Steps', ylabel='Accuracy')
    ax.yaxis.grid(True, linestyle="dotted")
    ax.xaxis.grid(True, linestyle="dotted")

    fig.savefig(output_path)
    #plt.show()
    print("Done")


def main():
  #filtered_list = fetch_results(base_path="/work/anlausch/ConceptBERT/output/finetuning/rw/1.0_1.0_2_10/", subdir="nl-adapter/")
  filtered_list = fetch_results(base_path="/work/anlausch/ConceptBERT/output/finetuning/rw/1.0_1.0_2_10/",
                               subdir="nl-adapter_tune_all")
  output_results_as_csv(filtered_list,
                        output_path="/work/anlausch/ConceptBERT/output/finetuning/rw/1.0_1.0_2_10/nl-adapter_tune_all/results_filtered.csv")
  filtered_list = fetch_results(base_path="/work/anlausch/ConceptBERT/output/finetuning/rw/1.0_1.0_2_10/",
                               subdir="nl-adapter_tune_all_quick_insight")
  output_results_as_csv(filtered_list,
                        output_path="/work/anlausch/ConceptBERT/output/finetuning/rw/1.0_1.0_2_10/nl-adapter_tune_all_quick_insight/results_filtered.csv")
  filtered_list = fetch_results(base_path="/work/anlausch/ConceptBERT/output/finetuning/omcs/", subdir="free-wo-nsp-adapter_tune_all/")
  output_results_as_csv(filtered_list, output_path="/work/anlausch/ConceptBERT/output/finetuning/omcs/free-wo-nsp-adapter_tune_all/results_filtered.csv")
  filtered_list = fetch_results(base_path="/work/anlausch/replant/bert/finetuning/poc_over_time/", subdir="wn_binary_16_longer/")
  output_results_as_csv(filtered_list, output_path="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/results_filtered.csv")
  filtered_list = fetch_results(base_path="/work/anlausch/replant/bert/finetuning/poc_over_time/", subdir="base_16_longer/")
  output_results_as_csv(filtered_list, output_path="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16_longer/results_filtered.csv")
  #output_results_as_csv(filtered_list, output_path="/work/anlausch/ConceptBERT/output/finetuning/omcs/free-wo-nsp-adapter_tune_all/results_filtered.csv")

if __name__=="__main__":
  main()