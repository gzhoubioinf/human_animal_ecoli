#!/usr/bin/env python
# coding: utf-8

# Author: Ge Zhou
# Email: ge.zhou@kaust.edu.sa


import numpy as np
def plot_confidence_interval(datalabel, report):
    tmp = []  # Temporary list to store labels with values <= 1

    # Extracting keys (labels) from datalabel where the value is <= 1
    for k in datalabel:
        if datalabel.get(k) <= 1:
            tmp.append(k)
    key = tmp.copy()
    # Re-arranging keys in 'key' dictionary based on values in datalabel
    for i in range(len(key)):
        key[datalabel[tmp[i]]] = tmp[i]

    # Defining flags and labels for metrics
    flag = ['precision', 'recall', 'f1-score']
    flag_mean = ['precision_mean', 'recall_mean', 'f1-score_mean']
    flag_std = ['precision_std', 'recall_std', 'f1-score_std']
    target_names = ['0', '1']
    target_type = {'0': key[0], '1': key[1]}

    # Dictionaries to store average, upper and lower values of metrics
    report_ave = {key[0]: [], key[1]: []}
    report_up = {key[0]: [], key[1]: []}
    report_low = {key[0]: [], key[1]: []}

    # Loop through target names to calculate metrics
    for name in target_names:
        val = {'precision': [], 'recall': [], 'f1-score': []}
        val_up = {'precision': [], 'recall': [], 'f1-score': []}
        val_low = {'precision': [], 'recall': [], 'f1-score': []}
        for fg in flag:
            a = []
            for ii in range(5):
                b = []
                for j in range(5):
                    b.append(report[ii * 5 + j][name][fg])
                a.append(np.max(b))
            # Calculating mean and standard deviation
            mn = np.mean(a)
            std = np.std(a)
            # Storing mean and confidence interval values
            val[fg].append(mn)
            val_up[fg].append(mn + 2.57 * std / np.sqrt(5))
            val_low[fg].append(mn - 2.57 * std / np.sqrt(5))
        report_ave[target_type[name]].append(val)
        report_up[target_type[name]].append(val_up)
        report_low[target_type[name]].append(val_low)

    # Checking the platform and storing the results in a file if on a Linux platform
    dataset = {
        'report_ave': report_ave,
        'report_up': report_up,
        'report_low': report_low
    }

    return dataset


