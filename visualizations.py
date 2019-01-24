# -*- coding: utf-8 -*-
###############################################################################
# Module:      visualizations
# Description: repo of tools for visualization
# Authors:     Yage Wang
# Created:     08.22.2018
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

def plot_performance(out_true,out_pred,save_name=None):
    """
    Plot ROC, Precision-Recall, Precision-Threshold
    @param out_true: list of output booleans indicicating if True
    @param out_pred: list of probabilities

    Examples
    --------
    >>> from enid.visualizations import plot_performance
    >>> y_true = [0, 1, 1, 0]
    >>> y_prob = [0.0000342, 0.99999974, 0.84367323, 0.5400342]
    >>> plot_performance(y_true, y_prob, None)
    """
    # get output
    precision,recall,thresholds = precision_recall_curve(out_true,out_pred)
    fpr,tpr,_  = roc_curve(out_true,out_pred)
    # roc
    fig = plt.figure(1,figsize=(18,3))
    plt.subplot(141)
    plt.plot(fpr,tpr, color='darkorange',lw=2,label='ROC (area = %0.3f)' % roc_auc_score(out_true,out_pred))
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title("ROC")
    plt.legend(loc="lower right")    
    
    # precision recall
    plt.subplot(142)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    plt.title('Precision Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot((0,1),(0.5,0.5),'k--')
    
    # precision threshold
    plt.subplot(143)
    plt.scatter(thresholds, precision[:-1], color='k',s=1)
    plt.title('Precision Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.plot((0,1),(0.5,0.5),'k--')
    
    # calibration
    plt.subplot(144)
    fraction_of_positives,mean_predicted_value = calibration_curve(out_true,out_pred,n_bins=5)
    plt.plot(mean_predicted_value,fraction_of_positives)
    plt.title('Calibration')
    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Fraction of Positives')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.grid()
    plt.plot((0,1),'k--')
    plt.show()
    if save_name: fig.savefig(save_name,bbox_inches='tight')
	
def plot_comparison(y_true,y_score_1,y_score_2,name_1,name_2,thre=0.5,save_name=None):
    """
    Plot ROC, Precision-Recall, Precision-Threshold
    @param y_true: list of output booleans indicicating if True
    @param y_score_1: list of probabilities of model 1
    @param y_score_2: list of probabilities of model 2
    @param name_1: name of model 1
    @param name_2: name of model 2
    @param thre: threshold for point marker on the curves

    Examples
    --------
    >>> from enid.visualizations import plot_comparison
    >>> y_true = [0, 1, 1, 0]
    >>> y_prob_1 = [0.0000342, 0.99999974, 0.84367323, 0.5400342]
    >>> y_prob_2 = [0.0000093, 0.99999742, 0.99999618, 0.2400342]
    >>> plot_comparison(y_true, y_prob_1, y_prob_2, 'SVC', 'XGBoost')
    """

    fig = plt.figure(1,figsize=(5,3))

    precision1,recall1,thresholds1 = precision_recall_curve(y_true, y_score_1)
    precision2,recall2,thresholds2 = precision_recall_curve(y_true, y_score_2)
    plt.step(recall1, precision1, label=name_1, color='b', alpha=0.5,where='post')
    plt.step(recall2, precision2, label=name_2, color='r', alpha=0.5,where='post')
    ppoint1 = precision1[:-1][np.argmin(np.abs(thresholds1 - thre))]
    rpoint1 = recall1[:-1][np.argmin(np.abs(thresholds1 - thre))]
    plt.plot(rpoint1, ppoint1, 'bo', markersize=7, label='thre'+str(thre))
    ppoint2 = precision2[:-1][np.argmin(np.abs(thresholds2 - thre))]
    rpoint2 = recall2[:-1][np.argmin(np.abs(thresholds2 - thre))]
    plt.plot(rpoint2, ppoint2, 'ro', markersize=7, label='thre'+str(thre))
    plt.title('PR Compare')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend()
    plt.show()
    if save_name: fig.savefig(save_name,bbox_inches='tight')

def plot_numerics(out_true, out_pred, log=False, save_name=None):
    """
    Plot trend lines, side-by-side histogram
    @param out_true: list of output numbers indicicating if True
    @param out_pred: list of predicted numbers

    Examples
    --------
    >>> from shakespeare.visualizations import plot_performance
    >>> y_true = [0, 1, 1, 0]
    >>> y_prob = [0.0000342, 0.99999974, 0.84367323, 0.5400342]
    >>> p
    """
    fig = plt.figure(1,figsize=(15,10))
    plt.style.use('seaborn-deep')

    # trends lines
    plt.subplot(211)
    sorted_y = sorted([(y, i) for i,y in enumerate(out_true)])
    sorted_y, idx = np.array([y[0] for y in sorted_y]), [y[1] for y in sorted_y]
    sorted_pred = out_pred[idx]
    if log: sorted_y, sorted_pred = np.log(sorted_y), np.log(sorted_pred)
    x = list(range(sorted_y.shape[0]))

    plt.plot(x, sorted_y, label='TRUE')
    plt.plot(x, sorted_pred, label='PRED', alpha=0.7)
    plt.title('Trend Compare')
    plt.xlabel('Exemplar Index (sorted by y_true)')
    if log: plt.ylabel('Value (log)')
    else:   plt.ylabel('Value')
    plt.legend()

    # hist
    plt.subplot(212)
    floor, ceil = np.min(np.concatenate([out_true, out_pred], axis=0)), np.max(np.concatenate([out_true, out_pred]))
    floor, ceil = int(np.floor(floor / 10.0)) * 10, int(np.ceil(ceil / 10.0)) * 10
    bins = np.linspace(floor, ceil, 30)
    plt.hist([out_true, out_pred], bins, label=['TRUE', 'PRED'])
    plt.title('Histogram')
    plt.legend()
    plt.show()
    if save_name: fig.savefig(save_name,bbox_inches='tight')