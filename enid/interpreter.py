# -*- coding: utf-8 -*-
###############################################################################
# Module:      Interpreter
# Description: Visualization module to explain Time-Aware Hierarchical Attention Model
# Authors:     Yage Wang
# Created:     5.1.2019
###############################################################################

import os
import re
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpld3
from .data_helper import HierarchicalVectorizer
from .than_clf import T_HAN

def monitor_interpreter(data, model_path, step=None, most_recent=20, save_name=None):
    """
    Interprete results for one example by re-applying model along time
    
    Parameters
    ----------
    data : dict
        one data point with format {time: [tokens]}. For example:
        {Timestamp('2013-01-09 00:00:00'): ['GPI-8120000000'],
         Timestamp('2013-01-10 00:00:00'): ['CPT-99212', 'POS-22', 'TOB-131', 'ICD9DX-V202'],
         ...
         Timestamp('2015-03-21 00:00:00'): ['GPI-6610002000', 'GPI-6420001000', 'GPI-4120003010']}

    model_path : str
        the path to store the model
    
    step : int, optional (defult None)
        if not None, load specific model with given step
    
    most_recent : int, optional (default 20)
        if provided, only plot most_recent timestampls
    
    save_name : str, optional (default None)
        the HTML file path to output interactive visualization; if None, just display

    Returns
    ----------
    
    """
    model = T_HAN('deploy', model_path=model_path, step=step)
    vec = HierarchicalVectorizer()
    most_recent = min(most_recent, len(data)-1)
    
    t_, x_ = [], []
    dates = sorted(list(data.keys()))
    for i in range(most_recent)[::-1]:
        t, x = vec({dates[idx]: data[dates[idx]] for idx in range(max(0, len(data)-i-most_recent), len(data)-i)},
                    model.max_sequence_length, model.max_sentence_length)
        t_.append(np.expand_dims(t, 0))
        x_.append(np.expand_dims(x, 0))
    t_, x_ = np.concatenate(t_, axis=0), np.concatenate(x_, axis=0)

    output = model.deploy(t_, x_)
    output = (output - output.min()) / (output.max() - output.min())
    del model
    most_recent = output.shape[0]
    data_to_show = {d:data[d] for d in dates[-most_recent:]}
    
    variables = pickle.load(open(os.path.join(os.path.dirname(__file__),'pickle_files','codes'), 'rb'))
    data_to_show = {k: [f"{c}  {str(variables.get(c, ''))}" for c in v if re.search(r'^(ICD|GPI|CPT|HCPCS)', c)] for k, v in data_to_show.items()}
    dates = list(data_to_show.keys())
    names = ['<h3>'+'<br>'.join([c if len(c)<100 else c[:97]+'...' for c in v])+'</h3>' for v in data_to_show.values()]
    
    cmap = plt.get_cmap('GnBu')
    sizes = [len(v)*75 for v in data_to_show.values()]
    
    mpld3.enable_notebook()
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=(int(most_recent/2), 2), dpi=120)
        
        # Create the base line
        start = min(dates)
        stop = max(dates)
        ext = int((mdates.date2num(stop) - mdates.date2num(start)) * 0.05)
        ax.plot((mdates.date2num(start)-ext, mdates.date2num(stop)+ext), (0, 0), 'k', alpha=.5)
        ax.plot(mdates.date2num(stop)+ext, 0, marker='>', lw=2, c='k', alpha=.5)
        scatter = ax.scatter(dates, [0]*most_recent, s=sizes, c=output, cmap=cmap,
                             vmin=0.0, vmax=1.0, edgecolor='gray', alpha=0.9)
        cbar = plt.colorbar(scatter, )
        cbar.set_label("Risk Level (%)")

        # Set the xticks formatting
        # format xaxis with 3 month intervals
        ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=1))
        ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
        
        # Remove components for a cleaner look
        plt.setp((ax.get_yticklabels() + ax.get_yticklines() +
                  list(ax.spines.values())), visible=False)
        
        tooltip = mpld3.plugins.PointHTMLTooltip(scatter, labels=names)
        mpld3.plugins.connect(fig, tooltip)
        if save_name: mpld3.save_html(fig, save_name)
        else: mpld3.show(fig)

def attention_interpreter(data, model_path, step=None, most_recent=20, save_name=None):
    """
    Interprete results for one example by re-applying model along time
    
    Parameters
    ----------
    data : dict
        one data point with format {time: [tokens]}. For example:
        {Timestamp('2013-01-09 00:00:00'): ['GPI-8120000000'],
         Timestamp('2013-01-10 00:00:00'): ['CPT-99212', 'POS-22', 'TOB-131', 'ICD9DX-V202'],
         ...
         Timestamp('2015-03-21 00:00:00'): ['GPI-6610002000', 'GPI-6420001000', 'GPI-4120003010']}

    model_path : str
        the path to store the model
    
    step : int, optional (defult None)
        if not None, load specific model with given step
    
    most_recent : int, optional (default 20)
        if provided, only plot most_recent timestampls
    
    save_name : str, optional (default None)
        the HTML file path to output interactive visualization; if None, just display

    Returns
    ----------
    
    """
    
    model = T_HAN('deploy', model_path=model_path, step=step)
    vec = HierarchicalVectorizer()
    most_recent = min(most_recent, len(data))
    
    token_att = model.graph.get_tensor_by_name("token_attention:0")
    sntnc_att = model.graph.get_tensor_by_name("sentence_attention:0")
    t_, x_ = vec(data, model.max_sequence_length, model.max_sentence_length)
    
    probs = model.deploy(np.expand_dims(t_, 0), np.expand_dims(x_, 0))
    tokens, sentences = model.sess.run([token_att, sntnc_att],
                                       {model.input_x: np.expand_dims(x_, 0),
                                        model.input_t: np.expand_dims(t_, 0)})
    
    tokens, x_, sentences = tokens[-most_recent:, :], x_[-most_recent:, :], sentences[0, -most_recent:]
    tokens = tokens * sentences.reshape([-1, 1])
    tokens[x_==vec.variable_size] = -1
    
    variables = pickle.load(open(os.path.join(os.path.dirname(__file__),'pickle_files','codes'), 'rb'))
    names = [[f"{vec.all_variables[c]}  {str(variables.get(vec.all_variables[c], ''))}" for c in l if c < vec.variable_size] for l in x_.tolist()]
    dates = list(data.keys())[-most_recent:]
    names = [[c if len(c)<100 else c[:97]+'...' for c in l] for l in names]
    names = list(itertools.chain(*[['<h3>'+ c +'</h3>' for c in l] for l in names]))
    
    tokens = tokens.T
    
    cmap = plt.get_cmap('GnBu')
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=(int(most_recent/2), int(model.max_sentence_length/4)), dpi=120)
        
        # Create the base line
        start = min(dates)
        stop = max(dates)
        ext = int((mdates.date2num(stop) - mdates.date2num(start)) * 0.05)
        ax.plot((mdates.date2num(start)-ext, mdates.date2num(stop)+ext), (0, 0), 'k', alpha=.5)
        ax.plot(mdates.date2num(stop)+ext, 0, marker='>', lw=2, c='k', alpha=.5)
        scatter = ax.scatter(list(itertools.chain(*[[d]*min(len(data[d]), model.max_sentence_length) for d in dates])),
                             list(itertools.chain(*[list(range(0, -min(len(data[d]), model.max_sentence_length), -1)) for d in dates])),
                             c=tokens[tokens>0.], cmap=cmap, marker='s',
                             s=100)#, vmin=0.0, vmax=1.0)
        del model

        # Set the xticks formatting
        # format xaxis with 3 month intervals
        plt.xlabel(f"Current Risk Level: {probs[0]}")
        ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=1))
        ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
        
        # Remove components for a cleaner look
        plt.setp((ax.get_yticklabels() + ax.get_yticklines() +
                  list(ax.spines.values())), visible=False)
        
        tooltip = mpld3.plugins.PointHTMLTooltip(scatter, labels=names)
        mpld3.plugins.connect(fig, tooltip)
        if save_name: mpld3.save_html(fig, save_name)
        else: mpld3.show(fig)