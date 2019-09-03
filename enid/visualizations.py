# -*- coding: utf-8 -*-
###############################################################################
# Module:      visualizations
# Description: repo of tools for visualization
# Authors:     Yage Wang
# Created:     08.22.2018
###############################################################################

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve


def plot_performance(out_true, out_pred, save_name=None):
    """
    Plot ROC, Precision-Recall, Precision-Threshold and Calibration

    Parameters
    --------
    out_true : list or 1-D array
        list of output booleans indicicating if True

    out_pred : list or 1-D array
        list of probabilities

    save_name : str, optional (default: None)
        the path of output image; if provided, save the plot to disk

    Examples
    --------
    >>> from enid.visualizations import plot_performance
    >>> y_true = [0, 1, 1, 0]
    >>> y_prob = [0.0000342, 0.99999974, 0.84367323, 0.5400342]
    >>> plot_performance(y_true, y_prob, None)
    """
    # get output
    precision, recall, thresholds = precision_recall_curve(out_true, out_pred)
    fpr, tpr, _ = roc_curve(out_true, out_pred)

    with plt.style.context('ggplot'):
        # roc
        fig = plt.figure(1, figsize=(15, 3))
        plt.subplot(141)
        plt.plot(
            fpr,
            tpr,
            color='seagreen',
            lw=2,
            label='ROC (area = %0.3f)' %
            roc_auc_score(
                out_true,
                out_pred))
        plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='seagreen')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('Recall')
        plt.title("ROC")
        plt.legend(loc="lower right")

        # precision recall
        plt.subplot(142)
        plt.step(recall, precision, color='dodgerblue', lw=2, where='post')
        plt.fill_between(
            recall,
            precision,
            step='post',
            alpha=0.2,
            color='dodgerblue')
        plt.title('Precision Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot((0, 1), (0.5, 0.5), 'k--')

        # precision threshold
        plt.subplot(143)
        plt.scatter(thresholds, precision[:-1], color='k', s=1)
        plt.title('Precision Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot((0, 1), (0.5, 0.5), 'k--')

        # calibration
        plt.subplot(144)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            out_true, out_pred, n_bins=5)
        plt.plot(
            mean_predicted_value,
            fraction_of_positives,
            label='Brier = %0.3f' %
            brier_score_loss(
                out_true,
                out_pred))
        plt.title('Calibration')
        plt.xlabel('Mean Predicted Value')
        plt.ylabel('Fraction of Positives')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.legend()
        plt.plot((0, 1), 'k--')
        plt.show()
        if save_name:
            fig.savefig(save_name, bbox_inches='tight')


def plot_numerics(out_true, out_pred, log=False, save_name=None):
    """
    Plot trend lines, side-by-side histogram

    Parameters
    --------
    out_true : list or 1-D array
        list of ground truth y

    out_pred : list or 1-D array
        list of y-hat

    log : boolean, optional (default: False)
        if True, tranform y-axis as logged for visualization purpose

    save_name : str, optional (default: None)
        the path of output image; if provided, save the plot to disk

    Examples
    --------
    >>> from shakespeare.visualizations import plot_performance
    >>> y_true = [0, 1, 1, 0]
    >>> y_prob = [0.0000342, 0.99999974, 0.84367323, 0.5400342]
    >>> plot_numerics(y_true, y_prob, log=True)
    """
    with plt.style.context('ggplot'):
        fig = plt.figure(1, figsize=(15, 10))

        # trends lines
        plt.subplot(211)
        x = list(range(out_true.shape[0]))
        if len(out_true) > 150:
            sorted_y = sorted([(y, i) for i, y in enumerate(out_true)])
            sorted_y, idx = np.array([y[0] for y in sorted_y]), [
                y[1] for y in sorted_y]
            sorted_pred = out_pred[idx]
            if log:
                sorted_y, sorted_pred = np.log(sorted_y), np.log(sorted_pred)

            plt.plot(x, sorted_pred, label='PRED', color='seagreen', alpha=0.7)
            plt.plot(x, sorted_y, label='TRUE', lw=2, color='dodgerblue')
            plt.title('Trend Compare')
            plt.xlabel('Exemplar Index (sorted by y_true)')
            if log:
                plt.ylabel('Value (log)')
            else:
                plt.ylabel('Value')
        else:
            if log:
                out_true, out_pred = np.log(out_true), np.log(out_pred)
            plt.plot(x, out_pred, label='PRED', color='seagreen')
            plt.plot(x, out_true, label='TRUE', color='dodgerblue')
            plt.title('Trend Compare')
            plt.xlabel('Exemplar Index')
            if log:
                plt.ylabel('Value (log)')
            else:
                plt.ylabel('Value')
        plt.legend()

        # hist
        plt.subplot(212)
        floor, ceil = np.min(np.concatenate([out_true, out_pred], axis=0)), np.max(
            np.concatenate([out_true, out_pred]))
        floor, ceil = int(np.floor(floor / 10.0)) * \
            10, int(np.ceil(ceil / 10.0)) * 10
        bins = np.linspace(floor, ceil, 30)

        xx = np.linspace(floor, ceil, 2000)
        kde_true = stats.gaussian_kde(out_true)
        kde_pred = stats.gaussian_kde(out_pred)

        plt.hist([out_true, out_pred], bins, normed=True, label=[
                 'TRUE', 'PRED'], color=['dodgerblue', 'seagreen'], alpha=0.6)
        plt.plot(xx, kde_true(xx), color='dodgerblue')
        plt.plot(xx, kde_pred(xx), color='seagreen')
        plt.xlabel('Bins')
        plt.ylabel('Normalized Frequency')
        plt.title('Histogram')
        plt.legend()
        plt.show()
        if save_name:
            fig.savefig(save_name, bbox_inches='tight')


def plot_comparison(
        y_true,
        y_score_1,
        y_score_2,
        name_1,
        name_2,
        thre=0.5,
        save_name=None):
    """
    Plot ROC comparison

    Parameters
    --------
    y_true : list or 1-D array
        list of output booleans indicicating if True

    y_score_1 : list or 1-D array
        list of probabilities of model 1

    y_score_2 : list or 1-D array
        list of probabilities of model 2

    name_1 : str
        name of model 1

    name_2 : str
        name of model 2

    thre : float, optional (default: 0.5)
        threshold for point marker on the curves

    save_name : str, optional (default: None)
        the path of output image; if provided, save the plot to disk

    Examples
    --------
    >>> from shakespeare.visualizations import plot_comparison
    >>> y_true = [0, 1, 1, 0]
    >>> y_prob_1 = [0.0000342, 0.99999974, 0.84367323, 0.5400342]
    >>> y_prob_2 = [0.0000093, 0.99999742, 0.99999618, 0.2400342]
    >>> plot_comparison(y_true, y_prob_1, y_prob_2, 'SVC', 'XGBoost')
    """
    with plt.style.context('ggplot'):
        fig = plt.figure(1, figsize=(9, 3))

        plt.subplot(121)
        fpr1, tpr1, _ = roc_curve(y_true, y_score_1)
        fpr2, tpr2, _ = roc_curve(y_true, y_score_2)
        plt.plot(
            fpr1,
            tpr1,
            color='dodgerblue',
            lw=2,
            label=f'{name_1} (AUC = %0.3f)' %
            roc_auc_score(
                y_true,
                y_score_1))
        plt.plot(
            fpr2,
            tpr2,
            color='seagreen',
            lw=2,
            label=f'{name_2} (AUC = %0.3f)' %
            roc_auc_score(
                y_true,
                y_score_2))
        plt.title('ROC Compare')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Recall')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend()

        plt.subplot(122)
        precision1, recall1, thresholds1 = precision_recall_curve(
            y_true, y_score_1)
        precision2, recall2, thresholds2 = precision_recall_curve(
            y_true, y_score_2)
        plt.step(
            recall1,
            precision1,
            label=name_1,
            color='dodgerblue',
            alpha=0.5,
            where='post')
        plt.step(
            recall2,
            precision2,
            label=name_2,
            color='seagreen',
            alpha=0.5,
            where='post')
        ppoint1 = precision1[:-1][np.argmin(np.abs(thresholds1 - thre))]
        rpoint1 = recall1[:-1][np.argmin(np.abs(thresholds1 - thre))]
        plt.plot(
            rpoint1,
            ppoint1,
            color='dodgerblue',
            marker='o',
            markersize=7,
            label='thre' +
            str(thre))
        ppoint2 = precision2[:-1][np.argmin(np.abs(thresholds2 - thre))]
        rpoint2 = recall2[:-1][np.argmin(np.abs(thresholds2 - thre))]
        plt.plot(
            rpoint2,
            ppoint2,
            color='seagreen',
            marker='o',
            markersize=7,
            label='thre' +
            str(thre))
        plt.title('PR Compare')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend()
        plt.show()
        if save_name:
            fig.savefig(save_name, bbox_inches='tight')
