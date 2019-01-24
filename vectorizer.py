# -*- coding: utf-8 -*-
###############################################################################
# Module:      vectorizers
# Description: repo of vectorize perticular sections
# Authors:     Yage Wang
# Created:     2018.05.18
###############################################################################

import os
import pickle
from datetime import datetime
import numpy as np

class Vectorizer(object):
    """
    Aim to vectorize EHR CCD data into event contaiers for further Deep Learning use.
    """
    def __init__(self):
        """
        Initialize a vectorizer to repeat use; load section variable spaces
        """
        self.all_variables = list(pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','all_variables'),"rb")))
        self.variable_size = len(self.all_variables)

    def fit_transform(self, seq, max_sequence_length, encounter_limit=None):
        """
        Transform claim sequence from enid.data_helper into event containers

        Parameters
        --------
        seq: JSON (dict) type object
            The parsed data from enid.data_helper
        
        max_sequence_length: int
            Fixed padding latest number of time buckets
        
        max_token_length: int
            Fixed padding number within one time bucket of one section

        Return
        --------
        T: numpy array, shape (num_timestamp,)
            All standardized time bucket numbers

        X: numpy array, shape (num_timestamp,)
            The index of each event based on each section variable space

        Examples
        --------
        >>> from enid.vectorizer import Vectorizer
        >>> vec = Vectorizer()
        >>> vec.fit_transform(ehr, 200)[0]
        array([84954, 85460, 85560, 85582, 85584, 85740, 85741, 85834, 85835,
               85880, 85884, 85926, 85950, 85951, 85962, 85968, 86132])
        
        >>> vec.fit_transform(ehr, 200)[1]
        array([[[  138,  1146,  1457, ..., 26494, 26494, 26494],
                [ 3974,  3974,  3974, ...,  3974,  3974,  3974],
                ...,
                [   41,    41,    41, ...,    41,    41,    41],
                [ 8579,  8579,  8579, ...,  8579,  8579,  8579]],
               [[26494, 26494, 26494, ..., 26494, 26494, 26494],
                [ 3974,  3974,  3974, ...,  3974,  3974,  3974],
                ...,
                [   41,    41,    41, ...,    41,    41,    41],
                [ 8579,  8579,  8579, ...,  8579,  8579,  8579]],
               ...,
               [[26494, 26494, 26494, ..., 26494, 26494, 26494],
                [ 3974,  3974,  3974, ...,  3974,  3974,  3974],
                ...,
                [    0,     3,     5, ...,    41,    41,    41],
                [ 8579,  8579,  8579, ...,  8579,  8579,  8579]],
               [[26494, 26494, 26494, ..., 26494, 26494, 26494],
                [ 3974,  3974,  3974, ...,  3974,  3974,  3974],
                ...,
                [   41,    41,    41, ...,    41,    41,    41],
                [   24,   151,   169, ...,  8579,  8579,  8579]]])
    
        >>> vec.fit_transform(ehr, 200)[2]
        array([[[ 1.62456340e-04,  1.62456340e-04,  1.62456340e-04, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                ...,
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],
               [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                ...,
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],
               ...,
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                ...,
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],
               [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                ...,
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,
                  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]])
        """
        T = [self._DT_standardizer(t) for t in seq['TIME']]
        X = []
        for x in seq['CODE']:
            try:             X.append(self.all_variables.index(x))
            except KeyError: continue
        if not T: return

        if encounter_limit: T = [t for t in T if t <= encounter_limit]
        T_delta = []
        for i, t in enumerate(T):
            if i == 0: T_delta.append(0)
            else: T_delta.append(t - T[i-1])

        T = np.array(T_delta, dtype='int32')
        X = np.array(X, dtype='int32')
        if T.shape[0] >= max_sequence_length:
            T = T[-max_sequence_length:]
            X = X[-max_sequence_length:]
        else:
            short_seq_length = max_sequence_length - T.shape[0]
            T = np.pad(T, (short_seq_length, 0), 'constant', constant_values=(0, 0))
            padding_values = np.array([self.variable_size] * short_seq_length)
            X = np.concatenate((padding_values, X), 0)

        return T, X

    ########################### PRIVATE FUNCTIONS #############################

    def _DT_standardizer(self, dt):
        if not dt: return None
        # use 1900-1-1 00:00:00 as base datetime; use time delta of base time to event time as rep
        std_dt = dt - datetime.strptime('01/01/1900', '%m/%d/%Y')
        # convert time delta from seconds to 12-hour bucket-size integer
        std_dt = int(std_dt.total_seconds() / 3600)
        if std_dt <= 0: return None
        else: return std_dt
