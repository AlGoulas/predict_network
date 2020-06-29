#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve

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
                     cv=None, scaler=None,
                     title='ROC'):
    # Logistic regression
    log_regr = LogisticRegression(max_iter=1000)
    
    all_tpr = []
    all_auc = []
    all_ap = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots()
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
        
        # Plot ROC and compute AUC
        viz = plot_roc_curve(log_regr, X_test, Y_test,
                             name='ROC fold {}'.format(i),
                             response_method='predict_proba',
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        
        # keep auc and tpr across folds
        all_tpr.append(interp_tpr)
        all_auc.append(viz.roc_auc) 
        
        disp_pr = plot_precision_recall_curve(log_regr, X_test, Y_test)
        disp_pr.ax_.set_title('Precision-Recall curve') 
        all_ap.append(disp_pr.average_precision)
    
    ax.set(title=title)
    
    return all_auc, all_ap
    
# Load the pickled data
name = 'data/net_data.pkl' 
net_data = load_data(name)

# Construct the dependent and independent variables
X, Y = create_x_y(dataset_name = 'Horvat_mouse', dataset = net_data)

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
cv = ShuffleSplit(n_splits=10, test_size=.3)

# Max Min scaler
scaler = MinMaxScaler()
 
# Predictions with logistic regression with current feature combination
all_combos_auc = []# Keep auc for all combos
all_combos_ap = []# Keep ap for all combos
for c in all_combos:
    title_names=''
    for n in c:
        title_names = title_names + ' ' + feature_names[n]
    
    all_auc, all_ap = cv_logistic_regr(X = X[:, c], Y = Y, 
                                       cv = cv, scaler = scaler,
                                       title = 'ROC ' + title_names)
    all_combos_auc.append(all_auc)
    all_combos_ap.append(all_ap)
    

            