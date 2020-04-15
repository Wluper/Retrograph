import sys
import jsonlines
import numpy as np

file_dataset = list(jsonlines.open(sys.argv[1]))

file_testresults = open(sys.argv[2], 'r').readlines()

assert len(file_dataset) == len(file_testresults)

print("Number of datapoints:", len(file_dataset))

acc = 0
for f_d, f_t in zip(file_dataset, file_testresults):
    if int(f_d['label']) == int(f_t.split(',')[1]):
        acc += 1

print("acc:", acc / len(file_dataset))
