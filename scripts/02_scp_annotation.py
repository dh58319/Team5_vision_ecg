#!/usr/bin/python3

import re
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import re

# Read the CSV files
meta = pd.read_csv('/home/donghyun/workbench/results/annotation_files/ptbxl_database.csv')
class_df = pd.read_csv('/home/donghyun/workbench/results/annotation_files/scp_statements.csv')

# Extract column names from class_df
column_names = class_df.iloc[:,0].values
column_names = np.sort(column_names)

result = pd.DataFrame(columns=column_names)
result_data = []

# make scp_codes file
for scp_code in meta['scp_codes']:
    values = re.findall(r'\d+\.\d+', scp_code)
    keys = re.findall(r'[A-Z]+', scp_code)
    result_dict = {}

    for key, value in zip(keys, values):
        result_dict[key] = value

    result_data.append(result_dict)

result = pd.DataFrame(result_data)
result = result.fillna("-")
result['ecg_id'] = meta['ecg_id']
result.to_csv('/home/donghyun/workbench/results/output_files/scp_codes.csv', index=False)

# make MI class only file
columns_with_mi = result.loc[:, result.columns.str.contains('MI')]

# plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,len(columns_with_mi.iloc[1,:]), figsize=(30,3))

for i in  range(len(columns_with_mi.iloc[1,:])):
    ax[i].hist(columns_with_mi.iloc[:,i])
    ax[i].set_xlabel(columns_with_mi.columns[i])

fig.savefig('/home/donghyun/workbench/results/output_files/distribution.png', bbox_inches='tight')

# save MI only file
columns_with_mi['ecg_id'] = result['ecg_id']
columns_with_mi.to_csv('/home/donghyun/workbench/results/output_files/MI_annotation.csv', index=False)