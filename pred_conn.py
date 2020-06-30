#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

def load_data(name):
    with open(name, 'rb') as fp:
        data = pickle.load(fp)
    return data    
 
def create_x_y(dataset_name=None, dataset=None):
    for ds in dataset:
        if ds['dataset'] == dataset_name: #dataset name is the desired one?
            C = ds['C']         #network matrix
            Dist = ds['Dist']   #distance matrix
            Delta = ds['Delta'] #delta matrix (cytology difference)
    idx = np.where(~np.eye(C.shape[0], dtype=bool)) #get the non-diagonal elements
    Y = C[idx] #dependent variable (connections in the network)
    X = np.vstack((Dist[idx], Delta[idx])) #predictors (distance and cytology)
    X = X.T
    
    return X, Y
    
def cv_logistic_regr(X=None, Y=None,
                     cv=None, scaler=None):
    # Logistic regression
    log_regr = LogisticRegression(max_iter=1000)
    
    all_tpr = []
    all_fpr = []
    all_auc = []
    
    all_prec = []
    all_recall = []
    all_ap = []
        
    for i, (train, test) in enumerate(cv.split(X, Y)):
        
        X_train, Y_train = X[train, :], Y[train]
        X_test, Y_test = X[test, :], Y[test]
        
        # If only one feature, then reshape
        if X.shape[1] == 1:
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)
    
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test) 
    
        log_regr.fit(X_train, Y_train)
        scores = log_regr.predict_proba(X_test)
        
        # ROC
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, 
                                                 scores[:, 1], 
                                                 pos_label=1)
        auc = metrics.auc(fpr, tpr)
        
        # keep auc fpr and tpr across folds
        all_tpr.append(tpr)
        all_fpr.append(fpr)
        all_auc.append(auc) 
        
        # Precision-recall
        precision, recall, thresholds = metrics.precision_recall_curve(Y_test, 
                                                                       scores[:, 1],
                                                                       pos_label=1)
        ap = metrics.average_precision_score(Y_test, 
                                             scores[:, 1], 
                                             pos_label=1)
       
        all_prec.append(precision)
        all_recall.append(recall)
        all_ap.append(ap)
    
    return all_auc, all_ap, all_tpr, all_fpr, all_prec, all_recall
  
def visualize_curve(x, y, 
                    metrics=None,
                    metric_name=None,
                    x_label=None,
                    y_label=None,
                    title=None):
    
    fig, ax = plt.subplots()
    for current_x, current_y in zip(x,y):
        ax.plot(current_x, current_y,
                lw=2, alpha=.8)
        ax.set_xlabel(x_label) 
        ax.set_ylabel(y_label)
        mean_metrics = np.mean(metrics)
        std_metrics = np.std(metrics)
        label_metrics='(' + metric_name + '= %0.2f $\pm$ %0.2f)' % (mean_metrics, std_metrics)
        ax.set_title(title + ' ' + label_metrics)

def visualize_data_frame(df=None, filters=None,
                         xlabel=None, ylabel=None,
                         file_name=None, path_save=None,
                         palette=None):
    
    # reduce the dataframe by keeping only the rows with the column
    # values specified in filters
    if filters is not None:
        for key, value in filters.items():
            df = df[(df[key] == value)]
    
    fig = plt.figure()
    fig.set_size_inches(10, 10)  
    
    sns.set(font_scale=2)
    sns.set_style('white') 
    
    ax = sns.boxplot(
                    x='grouping', 
                    y='values', 
                    data=df, 
                    palette=palette
                    )
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    # If a path is specififed, then save the figure as .svg
    if path_save is not None:
        file_name = file_name + '.svg'
        file_to_save= path_save / file_name
        plt.savefig(file_to_save, format='svg')
               
# Analyze the data
# Load the pickled data
name = 'data/net_data.pkl' 
net_data = load_data(name)

# Construct the dependent and independent variables from the desired dataset
# dataset_name = 'macaque_monkey' 'Horvat_mouse' 'marmoset_monkey'
dataset_name = 'macaque_monkey'
X, Y = create_x_y(dataset_name = dataset_name, dataset = net_data)

# Binarize Y
Y[np.where(Y!=0)[0]] = 1.

# Select observations for which for all features measurements exist (not nan)
sum_X = np.sum(X, axis=1)
idx_not_nan = ~np.isnan(sum_X)

X = X[idx_not_nan,:]
Y = Y[idx_not_nan]

# Normalize the predictors - take into account the abs valeus of 2nd predictor (cytology)
X = abs(X)

# Which features to use - calculate all possible combinations given the feat nr
feature_idx = [0, 1]
feature_names = ['dist', 'cyto']
all_combos = []
for l in range(1, len(feature_idx)+1):
    a = list(itertools.combinations(feature_idx, l))
    for item in a:
        all_combos.append(list(item))

# Monte Carlo CV
cv = ShuffleSplit(n_splits=100, test_size=.3)

# Max Min scaler
scaler = MinMaxScaler()
 
# Predictions with logistic regression with current feature combination
all_combos_auc = []# Keep auc for all combos
all_combos_ap = []# Keep ap for all combos
for c in all_combos:
    title_names=''
    for n in c:
        title_names = title_names + ' ' + feature_names[n]
    
    (all_auc, all_ap,
     all_tpr, all_fpr, all_prec, all_recall) = cv_logistic_regr(X = X[:, c], 
                                                                Y = Y, 
                                                                cv = cv, 
                                                                scaler = scaler
                                                               )
    all_combos_auc.append(all_auc)
    all_combos_ap.append(all_ap)
    
    # Visualize
    # ROC
    visualize_curve(all_fpr, all_tpr,
                    metrics=all_auc,
                    metric_name='AUC',
                    x_label='False Positive Rate',
                    y_label='True Positive Rate',
                    title= 'ROC ' + title_names)  
    
    # Precision-recall
    visualize_curve(all_recall, all_prec,
                metrics=all_ap,
                metric_name='AP',
                x_label='Recall',
                y_label='Precision', 
                title= 'Precision-recall ' + title_names)
    
#Visualize outside the lopp a summary of AUC and AP
# Boxplots of AUC
a = ['dist'] * len(all_combos_auc[0]) 
b = ['cyto'] * len(all_combos_auc[0])
c = ['dist+cyto'] * len(all_combos_auc[0])
category = a + b + c # This category labels is good for the rest of the boxplots

values = np.hstack((all_combos_auc[0], 
                   all_combos_auc[1],
                   all_combos_auc[2])
                   )

for_data_frame = {}
for_data_frame['values'] = values 
for_data_frame['grouping'] = category  

df = pd.DataFrame(for_data_frame) 
visualize_data_frame(df=df, filters=None, 
                     xlabel='predictors', ylabel='AUC',
                     file_name=None, path_save=None,
                     palette = sns.color_palette('mako_r', 3)
                    )  
# Boxplots of AP
values = np.hstack((all_combos_ap[0], 
                   all_combos_ap[1],
                   all_combos_ap[2])
                   )
for_data_frame['values'] = values 
 
df = pd.DataFrame(for_data_frame) 
visualize_data_frame(df=df, filters=None, 
                     xlabel='predictors', ylabel='AP',
                     file_name=None, path_save=None,
                     palette = sns.color_palette('mako_r', 3)
                    )  
            