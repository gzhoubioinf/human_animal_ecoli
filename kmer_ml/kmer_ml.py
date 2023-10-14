#!/usr/bin/env python
# coding: utf-8

# Author: Ge Zhou
# Email: ge.zhou@kaust.edu.sa



import os
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
#import matplotlib.pyplot as plt
import shap
#import importlib
#importlib.reload(shap)
#shap.initjs()
import sys
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime
from math import sqrt
import warnings
testflag = False

warnings.filterwarnings('ignore')

# # K-mer dataset processing

# ## Starting with reading chunk files(2000 in total)

# In[ ]:


# Function to retrieve all file names in a specified directory 
# that start with the string 'chunk' followed by a numerical value.
def file_name_listdir_local(file_dir):
    files_local = []  # Initialize an empty list to hold the file names.
    for files in os.listdir(file_dir):  # Loop through all files in the specified directory.
    #    print(files)  # (Optional) print the name of each file (line commented out).
        # Check if the file name starts with 'chunk' and the rest of the name is a numerical value.
        if files.startswith('chunk') and files[6:].isdigit():  
            files_local.append(files)  # If condition met, add the file name to the list.
    return files_local  # Return the list of file names meeting the specified criteria.


# ## Generating a Sparse Matrix and Establishing Cut-off Values (0.01, 0.05, 0.1...) for Filtering
# 

# In[ ]:


# Define a function named 'get_datamatrix' with parameters 'row_list', 'files_select', 'datapath', and 'cutoff'
def get_datamatrix(row_list,numb_files_select,datapath='./',cutoff=0.05):
    # Call file_name_listdir_local function to obtain a list of files from the specified directory
    files_local = file_name_listdir_local(datapath)

    # Restrict the list of files to the first 'files_select' number of files
        
    if numb_files_select > 0 and numb_files_select < len(files_local):
        files_local = files_local[0:numb_files_select]

    # Initialize an empty dictionary to store the column vocabulary
    vocabulary_col = {}

    # Initialize an empty dictionary to store the row vocabulary
    vocabulary_row = {}

    # Populate the row vocabulary with the values from 'row_list'
    for rl in row_list:
        _ = vocabulary_row.setdefault(rl, len(vocabulary_row))

    # Initialize counters and empty lists for further processing
    num_removed = 0
    num_preserved  = 0
    row =[]
    col = []
    data = []
    num_list = []  # This list is initialized but not used within the function

    # Iterate through each file in the restricted list of files
    for filename in files_local:
        # Open and read the file in binary mode
        with open(datapath + filename, 'rb') as f:
            ob = f.readlines()
            # Iterate through each line in the file
            for lines in ob:
                n = 0
                linestr = lines.decode('utf-8').split()  # Decode the binary line to utf-8 and split it into words

                # Count the occurrences of words starting with 'assembly' and ending with a digit
                for s in linestr:
                    if s.startswith('assembly'):
                        n += int(s[-1])

                # Check if the ratio of the count to 857.0 is greater than the cutoff value
                if n/857.0 > cutoff:
                    num_preserved += 1  # Increment the count of preserved lines
                    indx_col = vocabulary_col.setdefault(linestr[0], len(vocabulary_col))  # Update the column vocabulary

                    # Process each word in the line
                    for s in linestr:
                        if s.startswith('assembly'):
                            sp = s.split(':')
                            if sp[0][9:] in vocabulary_row:
                                col.append(indx_col)  # Append the column index
                                indx_row = vocabulary_row.get(sp[0][9:])  # Get the row index from the vocabulary
                                row.append(indx_row)  # Append the row index
                                data.append(int(sp[1]))  # Append the data value
                else:
                    num_removed += 1  # Increment the count of removed lines

    # Calculate the percentage of removed lines
    remove_percent = num_removed/(num_preserved + num_removed)

    # Create a Compressed Sparse Row matrix from the data, row, and col lists
    mtr = csr_matrix((data, (row, col)))

    # Return the sparse matrix, column vocabulary, row vocabulary, and removal percentage as output
    return mtr, vocabulary_col, vocabulary_row, remove_percent


# ## Get and Filter Data: Acquire the specified labels from the dataset and filter out the two designated data types.
#    ### HH&HA HH&AA AND HA&AA

# In[12]:


def get_datafilter(datalabel, filefold):
    # Reading data from a CSV file and storing it in a DataFrame called df
    if not filefold.endswith('/'):
        filefold = filefold +'/'
    df = pd.read_csv(filefold+'shortened_traits_scoary.csv')

    # Extracting values from the input dictionary datalabel and storing them in variables HA, HH, and AA
    HA = datalabel['HA']
    HH = datalabel['HH']
    AA = datalabel['AA']

    # Creating a new column called "data_type" in df.
    # This column is populated based on the values in the "mixed_other", "human_other", and "animal_other" columns of df.
    df["data_type"] = df.apply(lambda row: HA if row["mixed_other"] == 1 
                               else (HH if row["human_other"] == 1 
                                     else (AA if row["animal_other"] == 1 else None)), axis=1)

    # Filtering df to include only the rows where "data_type" is 0 or 1.
    # Storing the result in a new DataFrame called filtered1_df.
    filtered1_df = df[df['data_type'].isin([0, 1])]
    filtered1_df= filtered1_df.set_index('Name')

    # Returning the filtered DataFrame.
    return filtered1_df



def gridresearch_kfold(X,y,feature_names):
    # Initialize empty lists to store the performance, method, report, roc_auc,
    # best features, best feature indices, and best parameters
    performance = []
    method = []
    report = []
    roc_auc = []
    bestfeature =[]
    bestfeature_indices=[]
    bestpara = []

    # Run the process 5 times with different train-test splits
    for i in range(5):
        print(f"Train-Test Split {i+1}")
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Create a KFold object to split the training set into 5 folds
        kfold = KFold(n_splits=5, shuffle=True)

        # Loop through each fold
        for fold, (train_index, test_index) in enumerate(kfold.split(X_train, y_train), 1):
            # Split the training set into training and validation folds
            X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

            # Define the parameters and the model for grid search
            xgb_params = {'max_depth': [1, 3],'n_estimators': [20, 50],'subsample': [0.8, 1.0]}
            xgb_model = xgb.XGBClassifier(objective='binary:logistic')

            # Initialize GridSearchCV
            gs_clf = GridSearchCV(estimator=xgb_model, param_grid=xgb_params, cv=3, n_jobs=-1, scoring='balanced_accuracy')
            # Fit the model on the training fold
            gs_clf.fit(X_train_fold, y_train_fold)

            # Predict on the validation and test sets
            y_pred = gs_clf.predict(X_val_fold)
            pred_class_xgboost = gs_clf.predict(X_test)
            pred_probs_xgboost = gs_clf.predict_proba(X_test)

            # Get the best estimator and parameters from the grid search
            best_estimator = gs_clf.best_estimator_
            best_para = gs_clf.best_params_
            bestpara.append(best_para)

            # Get the feature importances from the best estimator
            best_features = gs_clf.best_estimator_.feature_importances_

            # Get the indices of the best features sorted in descending order
            best_feature_indices = np.argsort(best_features)[::-1]
            # Get the names of the best features
            best_feature_names = [feature_names[i] for i in best_feature_indices]
            bestfeature.append(best_feature_names)
            bestfeature_indices.append(best_feature_indices)

            print('best_featrure', best_feature_names[0:5])
            # Get the classification report for the validation fold
            report0 = classification_report(y_val_fold, y_pred, output_dict=True)

            # Get the balanced accuracy score for the validation fold
            score = balanced_accuracy_score(y_val_fold, y_pred)
            performance.append(score)

            report.append(report0)

            print("Best hyperparameters for this train-test split:")
            print(gs_clf.best_params_)
            print(score)
            print("Confusion matrix for this fold:")
            print(confusion_matrix(y_val_fold, y_pred))

    # If running on a Linux platform, save the results to a file

    dataset = {
        'performance':performance,
        'method':method,
        'report':report,
        'roc_auc':roc_auc,
        'bestfeature':bestfeature,
        'bestfeature_indices':bestfeature_indices,
        'bestpara':bestpara
    }
    return dataset
    # current_datetime = datetime.now()
    # datetime_string = current_datetime.strftime('%Y%m%d_%H%M')
    #
    # filename = f"gridreasearch_{datetime_string}.pickle"
    # with open(filename,'wb') as f:
    #     pickle.dump(dataset, f)
    #
    # # Return all the stored results
    # return  performance, method, report, roc_auc, bestfeature,bestfeature_indices,bestpara




# shap value 
def get_shapvalue(X, y, sub_bestfeature_indices,sub_bestfeature_name, best_para,class_name):
    X_subset = X[:,sub_bestfeature_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2)

    # Create and train your best XGBoost model
    best_model = xgb.XGBClassifier( objective='binary:logistic', 
                                  param_grid = best_para )

    #gs_clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=3, n_jobs=-1, scoring='balanced_accuracy')
    best_model.fit(X_train, y_train)

    best_model = xgb.XGBClassifier( objective='binary:logistic', 
                                      param_grid = best_para )

    #gs_clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=3, n_jobs=-1, scoring='balanced_accuracy')
    best_model.fit(X_train, y_train)

    tmp =  []
    datalabel = class_name
    for k in datalabel:
        if datalabel.get(k) <=1:
            tmp.append(k)
    key = tmp.copy()
    for i in range(len(key)):
        key[datalabel[tmp[i]]] = tmp[i]
    exp_set = X_subset.copy()
    
    ## K sample
    ksmp =100
    background_summary = shap.sample(exp_set, ksmp)

    explainer = shap.KernelExplainer(best_model.predict, background_summary)

    shap_values = explainer.shap_values(exp_set)   
    
    # data save
    dataset = {
        'shap_values':shap_values,
        'exp_set':exp_set,
        'sub_bestfeature_name':sub_bestfeature_name,
        'sub_bestfeature_indices':sub_bestfeature_indices,
        'key':key
    }
    #
    # current_datetime = datetime.now()
    # datetime_string = current_datetime.strftime('%Y%m%d_%H%M')
    #
    # filename = f"shapeKsmp{ksmp}_{datetime_string}.pickle"
    # with open(filename,'wb') as f:
    #     pickle.dump(dataset, f)

    return dataset

