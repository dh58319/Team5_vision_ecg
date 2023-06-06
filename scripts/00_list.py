#!/usr/bin/python3

import os
import numpy as np
import pandas as pd
import wfdb
import ast
import matplotlib.pyplot as plt
import csv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_path", required=True)
ap.add_argument("-o", "--output_path", required=True)
ap.add_argument("-s", "--step_size", type=int, default=100)
args = vars(ap.parse_args())

record_path = args['input_path']
output_path = args['output_path']
patient_fold = os.listdir(record_path)

fold_path = []
pid = []

for pf in patient_fold:
    path = os.path.join(record_path, pf)
    pid_list = list(set([e.split('_')[0] for e in os.listdir(path)]))
    pid.extend(pid_list)
    fold_path.extend([path] * len(pid_list))

with open(os.path.join(args['output_path'],"make_plot.log"), "w") as f:
    for i in range(0, len(pid)+1, args['step_size']):
        f.write(f"python3 /home/donghyun/workbench/scripts/01_plot.py -s {i} -e {i+args['step_size']}\n")

with open(os.path.join(args['output_path'],'fold_path.csv'), "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(fold_path)

with open(os.path.join(args['output_path'],'pid_path.csv'), "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(pid)

