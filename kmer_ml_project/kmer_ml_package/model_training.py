#!/usr/bin/env python
# coding: utf-8

# Author: Ge Zhou
# Email: ge.zhou@kaust.edu.sa

def gridresearch_kfold(X ,y ,feature_names):
    # Initialize empty lists to store the performance, method, report, roc_auc,
    # best features, best feature indices, and best parameters
    performance = []
    method = []
    report = []
    roc_auc = []
    bestfeature =[]
    bestfeature_indices = []
    bestpara = []

    # Run the process 5 times with different train-test splits
    for i in range(5):
        print(f"Train-Test Split {i + 1}")
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
            xgb_params = {'max_depth': [1, 3], 'n_estimators': [20, 50], 'subsample': [0.8, 1.0]}
            xgb_model = xgb.XGBClassifier(objective='binary:logistic')

            # Initialize GridSearchCV
            gs_clf = GridSearchCV(estimator=xgb_model, param_grid=xgb_params, cv=3, n_jobs=-1,
                                  scoring='balanced_accuracy')
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
            best_feature_names = [feature_names[best_feature_indices[i]] for i in range(1000)]
            bestfeature.append(best_feature_names)
            bestfeature_indices.append(best_feature_indices[0:1000])

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
    if sys.platform.startswith('linux'):
        dataset = {
            'performance': performance,
            'method': method,
            'report': report,
            'roc_auc': roc_auc,
            'bestfeature': bestfeature,
            'bestfeature_indices': bestfeature_indices,
            'bestpara': bestpara
        }

        current_datetime = datetime.now()
        datetime_string = current_datetime.strftime('%Y%m%d_%H%M')

        filename = f"gridreasearch_{datetime_string}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)

    # Return all the stored results
    return performance, method, report, roc_auc, bestfeature, bestfeature_indices, bestpara


#

# ## Calculating and Storing Confidence Intervals for Classification Metrics

# In[ ]:


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

    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime('%Y%m%d_%H%M')

    filename = f"confidence_interval_{datetime_string}.pickle"
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    return


# ## SHAP Value Calculation for XGBoost Model

# In[ ]:


# shap value
def get_shapvalue(X, y, sub_bestfeature_indices, sub_bestfeature_name, best_para, class_name):
    X_subset = X[:, sub_bestfeature_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2)

    # Create and train your best XGBoost model
    best_model = xgb.XGBClassifier(objective='binary:logistic',
                                   param_grid=best_para)

    # gs_clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=3, n_jobs=-1, scoring='balanced_accuracy')
    best_model.fit(X_train, y_train)

    best_model = xgb.XGBClassifier(objective='binary:logistic',
                                   param_grid=best_para)

    # gs_clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=3, n_jobs=-1, scoring='balanced_accuracy')
    best_model.fit(X_train, y_train)

    tmp = []
    datalabel = class_name
    for k in datalabel:
        if datalabel.get(k) <= 1:
            tmp.append(k)
    key = tmp.copy()
    for i in range(len(key)):
        key[datalabel[tmp[i]]] = tmp[i]
    exp_set = X_subset.copy()

    ## K sample
    ksmp = 10
    background_summary = shap.sample(exp_set, ksmp)

    explainer = shap.KernelExplainer(best_model.predict, background_summary)

    shap_values = explainer.shap_values(exp_set)

    # data save
    dataset = {
        'shap_values': shap_values,
        'exp_set': exp_set,
        'sub_bestfeature_name': sub_bestfeature_name
    }

    current_datetime = datetime.now()
    datetime_string = current_datetime.strftime('%Y%m%d_%H%M')

    filename = f"shapeKsmp{ksmp}_{datetime_string}.pickle"
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

    return shap_values
