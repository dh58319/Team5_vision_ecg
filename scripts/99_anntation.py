#!/usr/bin/python3

import argparse
import pandas as pd
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file")
ap.add_argument("-c", "--class")
args = vars(ap.parse_args())

annot = pd.read_csv(args['file'], index_col=0)
annot_mi = annot[[col for col in annot.columns if args['class'] in col]]
annot_mi['non'] = np.where(annot_mi.isna().all(axis=1), 0, 1)
annot_mi.to_csv('/home/donghyun/workbench/results/output_files/'+args['class']+'_annotation.csv')

# plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,len(annot_mi.iloc[1,:]), figsize=(30,3))

for i in  range(len(annot_mi.iloc[1,:])):
    ax[i].hist(annot_mi.iloc[:,i])
    ax[i].set_xlabel(annot_mi.columns[i])

fig.savefig('/home/donghyun/workbench/results/output_files/distribution.png', bbox_inches='tight')