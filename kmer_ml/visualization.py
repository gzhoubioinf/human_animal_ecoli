#!/usr/bin/env python
# coding: utf-8

# Author: Ge Zhou
# Email: ge.zhou@kaust.edu.sa


##SHAP values visualization

import pickle as pkl

# Load the pickle file
#example:result_kmer_cutoff0.01_file5_20230925_0008.pkl
filename = "shapeKsmp10_20230925_0008.pkl"
with open(filename, "rb") as f:
    dataset = pkl.load(f)
shap_values = dataset['shap_values']
exp_set = dataset['exp_set']
sub_bestfeature_name = dataset['sub_bestfeature_name']

#%%

shap.summary_plot(shap_values, exp_set,
                  feature_names=sub_bestfeature_name,
                  class_names=key,
                  show=False)
# Set the y-tick labels as numbers
num_ticks = len(sub_bestfeature_indices)
# plt.yticks( range(num_ticks))

plt.yticks(range(num_ticks), range(num_ticks))

# Print correspondence between numbers and feature names
for i, name in enumerate(sub_bestfeature_name[0:min(10,len(sub_bestfeature_name))]):
    print(f"{i}: {name}")
plt.tight_layout()
plt.show()

import numpy as np
# This is for type "HA" and "AA" visualization
flag = ['precision', 'recall', 'f1-score']
flag_mean = ['precision_mean', 'recall_mean', 'f1-score_mean']
flag_std = ['precision_std', 'recall_std', 'f1-score_std']
target_names = ['HA', 'AA']
class_label = {'HA': 1, 'HH': 2, 'AA': 0}
tn_num = {'HA': '1', 'AA': '0'}
report_ave = {'HA': [], 'AA': []}
report_up = {'HA': [], 'AA': []}
report_low = {'HA': [], 'AA': []}
for name in target_names:
    val = {'precision': [], 'recall': [], 'f1-score': []}
    val_up = {'precision': [], 'recall': [], 'f1-score': []}
    val_low = {'precision': [], 'recall': [], 'f1-score': []}
    for fg in flag:
        a = []
        for ii in range(5):
            b = []
            for j in range(5):
                b.append(report[ii * 5 + j][tn_num[name]][fg])
            a.append(np.max(b))
        mn = np.mean(a)
        std = np.std(a)
        val[fg].append(mn)
        val_up[fg].append(mn + 2.57 * std / np.sqrt(5))
        val_low[fg].append(mn - 2.57 * std / np.sqrt(5))
    report_ave[name].append(val)
    report_up[name].append(val_up)
    report_low[name].append(val_low)

# %%
import numpy as np
import matplotlib.pyplot as plt

# Plotting for the training performance
x = np.arange(2)
width = 0.2

fig, ax = plt.subplots()

for i, fg in enumerate(flag):
    vals = [report_ave[name][0][fg][0] for name in {'HA', 'AA'}]
    errs = [
        (report_ave[name][0][fg][0] - report_low[name][0][fg][0],
         report_up[name][0][fg][0] - report_ave[name][0][fg][0])
        for name in {'HA', 'AA'}]

    ax.bar(x + i * width, vals, width, label=fg, yerr=np.transpose(errs))

ax.set_ylabel('Scores')
ax.set_title('Performance Metrics (XGBoost_kmer)')
ax.set_xticks(x + width)
ax.set_xticklabels({'HA', 'AA'})

ax.legend()

plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

# Plotting
x = np.arange(2)
width = 0.2

fig, ax = plt.subplots()

for i, fg in enumerate(flag):
    vals = [report_ave[name][0][fg][0] for name in {'HA', 'AA'}]
    errs = [
        (report_ave[name][0][fg][0] - report_low[name][0][fg][0],
         report_up[name][0][fg][0] - report_ave[name][0][fg][0])
        for name in {'HA', 'AA'}]

    ax.bar(x + i * width, vals, width, label=fg, yerr=np.transpose(errs))

ax.set_ylabel('Scores')
ax.set_title('Performance Metrics (XGBoost_kmer)')
ax.set_xticks(x + width)
ax.set_xticklabels(['HA', 'AA'])
ax.legend()

plt.show()



