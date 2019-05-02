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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
import mpld3
from .data_helper import HierarchicalVectorizer
from .than_clf_gru import T_HAN

def monitor_interpreter(data, model_path, step=None, most_recent=20):
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
    del model
    most_recent = output.shape[0]
    data_to_show = {d:data[d] for d in dates[-most_recent:]}
    
    variables = pickle.load(open(os.path.join(os.path.dirname(__file__),'pickle_files','codes'), 'rb'))
    data_to_show = {k: [f"{c}  {str(variables.get(c, ''))}" for c in v if re.search(r'^(ICD|GPI|CPT|HCPCS)', c)] for k, v in data_to_show.items()}
    dates = list(data_to_show.keys())
    names = ['<h3>'+'<br>'.join([c if len(c)<50 else c[:47]+'...' for c in v])+'</h3>' for v in data_to_show.values()]
    
    cmap = plt.get_cmap('hot')
    normalize = Normalize(vmin=output.min(), vmax=output.max())
    colors = [cmap(normalize(value)) for value in output]
    sizes = list(output * 800)
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=(int(most_recent/2), 2), dpi=120)
        #levels = np.array([-9, 9, -6, 6, -3, 3])
        
        # Create the base line
        start = min(dates)
        stop = max(dates)
        ax.plot((start, stop), (0, 0), 'k', alpha=.5)
        scatter = ax.scatter(dates, [0]*most_recent, s=sizes, c=colors, zorder=9999)

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
        mpld3.show()
