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
ap.add_argument("-s", "--start", type=int, default=0)
ap.add_argument("-e", "--end", type=int, default=101)
args = vars(ap.parse_args())

record_path = '/home/donghyun/workbench/record_files/records500'
output_path = '/home/donghyun/workbench/results/pngfiles'
patient_fold = os.listdir(record_path)

with open("/home/donghyun/workbench/results/output_files/fold_path.csv", "r", newline="") as file:
    reader = csv.reader(file)
    fold_path = next(reader)

with open("/home/donghyun/workbench/results/output_files/pid.csv", "r", newline="") as file:
    reader = csv.reader(file)
    pid = next(reader)

for k in range(args['start'], args['end']):
    # load data
    data = wfdb.rdrecord(os.path.join(fold_path[k], pid[k]+'_hr'))
    data.p_signal = data.p_signal[:, 1:]
    
    # plot
    fig, ax = plt.subplots(11, 1, figsize=(20, 20))
    
    # Remove box lines
    for a in ax:
        a.spines['top'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.set_xticks([])
        a.set_yticks([])
        a.set_xticklabels([])
        a.set_yticklabels([])
    
    for i in range(11):
        ax[i].plot(data.p_signal[:, i])
    fig.savefig(os.path.join(output_path, pid[k] + '.png'), bbox_inches='tight')
    
