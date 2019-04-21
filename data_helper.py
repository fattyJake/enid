# -*- coding: utf-8 -*-
###############################################################################
# Module:      data_helper
# Description: repo of database functions for enid
# Authors:     Yage Wang
# Created:     08.10.2018
###############################################################################

import os
import random
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import re
import pyodbc
import pandas as pd

class Vectorizer(object):
    """
    Aim to vectorize claim data into event contaiers for further Deep Learning use.
    """
    def __init__(self, mode='cd'):
        """
        Initialize a vectorizer to repeat use; load section variable spaces
        """
        assert mode in ['cd', 'more2'], 'AttributeError: mode only acccept "cd" or "more2", got {}'.format(str(mode))
        if mode == 'cd':    self.all_variables = list(pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','all_variables'),"rb")))
        if mode == 'more2': self.all_variables = list(pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','all_variables_more2'),"rb")))
        self.variable_size = len(self.all_variables)

    def __call__(self, seq, max_sequence_length, encounter_limit=None):
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
        """
        T = [self._DT_standardizer(t) for t in seq['TIME']]
        X, removals = [], []
        for i,x in enumerate(seq['CODE']):
            try:               X.append(self.all_variables.index(x))
            except ValueError: removals.append(i)
        for i in sorted(removals, reverse=True): T.pop(i)
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
        std_dt = int(std_dt.total_seconds() / 3600 / 24)
        if std_dt <= 0: return None
        else: return std_dt

class HierarchicalVectorizer(object):
    """
    Aim to vectorize claim data into two-level event contaiers for further HAN use.
    """
    def __init__(self, mode='more2'):
        """
        Initialize a vectorizer to repeat use; load section variable spaces
        """
        assert mode in ['cd', 'more2'], 'AttributeError: mode only acccept "cd" or "more2", got {}'.format(str(mode))
        if mode == 'cd':    self.all_variables = list(pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','all_variables'),"rb")))
        if mode == 'more2': self.all_variables = list(pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','all_variables_more2'),"rb")))
        self.variable_size = len(self.all_variables)

    def __call__(self, seq, max_sequence_length, max_sentence_length):
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
        """
        seq = list(zip(seq['TIME'], seq['CODE']))
        seq = [(self._DT_standardizer(i[0]), self.all_variables.index(i[1])) for i in seq if i[1] in self.all_variables]
        grp_seq = {}
        for t, c in seq:
            d = grp_seq.setdefault(t, [])
            d.append(c)
        T, X = list(grp_seq.keys()), list(grp_seq.values())

        T_delta = []
        for i, t in enumerate(T):
            if i == 0: T_delta.append(0)
            else: T_delta.append(t - T[i-1])

        X = [random.sample(line, max_sentence_length) if len(line)>=max_sentence_length else line+[self.variable_size]*(max_sentence_length-len(line)) for line in X]

        T = np.array(T_delta, dtype='int32')
        X = np.array(X, dtype='int32')
        if T.shape[0] >= max_sequence_length:
            T = T[-max_sequence_length:]
            X = X[-max_sequence_length:, :]
        else:
            short_seq_length = max_sequence_length - T.shape[0]
            T = np.pad(T, (short_seq_length, 0), 'constant', constant_values=(0, 0))
            padding_values = np.array([[self.variable_size] * max_sentence_length] * short_seq_length)
            X = np.concatenate((padding_values, X), 0)

        return T, X

    ########################### PRIVATE FUNCTIONS #############################

    def _DT_standardizer(self, dt):
        if not dt: return None
        # use 1900-1-1 00:00:00 as base datetime; use time delta of base time to event time as rep
        std_dt = dt - datetime.strptime('01/01/1900', '%m/%d/%Y')
        # convert time delta from seconds to 12-hour bucket-size integer
        std_dt = int(std_dt.total_seconds() / 3600 / 24)
        if std_dt <= 0: return None
        else: return std_dt

def get_members(payer, server='CARABWDB03', date_start=None, date_end=None):
    """
    Retrieve memberIDs for further training
    @param payer: the name of the payer table
    @param date_start: YYYY-MM-DD
    @param date_end: YYYY-MM-DD

    Return
    --------
    List of client memberIDs

    Examples
    --------
    >>> from enid.fetch_db import get_members
    >>> get_members("CD_AHM", '2016-01-01', '2016-01-05')
    [1915417,
     1915416,
     ...
     1869173,
     1869172]
    """
    # initialize
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER='+server+';').cursor()
    date_start, date_end, = str(date_start), str(date_end)
    sql = """SELECT mem_id, mem_ClientMemberID
                   FROM """+payer+""".dbo.tbMember
                   WHERE mem_ClientMemberID IS NOT NULL AND dateInserted BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
                   ORDER BY mem_id DESC
                   """
    sql = re.sub("AND dateInserted BETWEEN 'None' AND 'None'", "", sql)
    cursor.execute(sql)
    return list(set([(i[0], i[1]) for i in cursor]))

def slip_trip_stumble_fall(memberID, payer='CD_HEALTHFIRST',server='CARABWDB03', date_start=None, date_end=None, mode='cd'):
    # initialize
    assert mode in ['cd', 'more2'], 'AttributeError: mode only acccept "cd" or "more2", got {}'.format(str(mode))
    if mode == 'cd':    falls_logic = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'falls_logic'), 'rb'))
    if mode == 'more2': falls_logic = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'falls_logic_more2'), 'rb'))
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER='+server+';').cursor()
    date_start, date_end = str(date_start), str(date_end)

    sql = """
        SELECT e.mem_id AS MemberID, e.enc_ID AS EncounterID, CASE icdVersionInd WHEN 9 THEN 'ICD9' WHEN 10 THEN 'ICD10' ELSE 'ICD9' END AS CodeType, ed.icd_Code AS Code
        FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
        INNER JOIN """+payer+""".dbo.tbEncounterDiagnosis ed ON e.enc_id = ed.enc_id
        WHERE e.mem_id = """+str(memberID)+""" AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    """

    sql = re.sub(r" AND (e\.)?[a-zA-Z\_]+ BETWEEN 'None' AND 'None'", "", sql)
    data = cursor.execute(sql).fetchall()
    one_year_codes = set([i[2]+'-'+i[3] for i in data])
    if one_year_codes & set(falls_logic): return 1
    else: return 0

def emergency_department(memberID, payer, server, last_day):
    ed_logic = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'ed_logic'), 'rb'))

    # initialize
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER='+server+';').cursor()
    date_start, date_end = (last_day+relativedelta(days=1)).strftime('%Y-%m-%d'), (last_day+relativedelta(months=6)).strftime('%Y-%m-%d')

    sql = """
    SELECT mem_id AS MemberID, enc_ID AS EncounterID, 'POS' AS CodeType, pos_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
    WHERE mem_id = """+str(memberID)+""" AND pos_Code = 23 AND enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT e.mem_id AS MemberID, e.enc_ID AS EncounterID, 'CPT' AS CodeType, eCPT.cpt_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterCPT eCPT ON e.enc_id = eCPT.enc_id
    WHERE e.mem_id = """+str(memberID)+""" AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT e.mem_id AS MemberID, e.enc_ID AS EncounterID, 'UBREV' AS CodeType, CASE LEN(eREV.rev_Code) WHEN 3 THEN '0'+eREV.rev_Code ELSE eREV.rev_Code END AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterRevenue eREV ON e.enc_id = eREV.enc_id
    WHERE e.mem_id = """+str(memberID)+""" AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT e.mem_id AS MemberID, e.enc_ID AS EncounterID, CASE icdVersionInd WHEN 9 THEN 'ICD9' WHEN 10 THEN 'ICD10' ELSE 'ICD9' END AS CodeType, ed.icd_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterDiagnosis ed ON e.enc_id = ed.enc_id
    WHERE e.mem_id = """+str(memberID)+""" AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT e.mem_id AS MemberID, e.enc_ID AS EncounterID, 'ICD10Proc' AS CodeType, eProc.icd_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterProcedure eProc ON e.enc_id = eProc.enc_id
    WHERE e.mem_id = """+str(memberID)+""" AND icdVersionInd = 10 AND e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    """

    sql = re.sub(r" AND (e\.)?[a-zA-Z\_]+ BETWEEN 'None' AND 'None'", "", sql)
    data = cursor.execute(sql).fetchall()
    halfyear_codes = list(set([(i[1], i[2]+'-'+i[3]) for i in data]))

    if not halfyear_codes: return 0
    halfyear_codes = pd.DataFrame(halfyear_codes)
    halfyear_codes.columns = ['ENC_ID', 'CODE']
    halfyear_codes = halfyear_codes.groupby('ENC_ID')['CODE'].agg(list).reset_index()
    halfyear_codes['main_1'] = halfyear_codes['CODE'].map(lambda x: True if list(set(x) & set(ed_logic['ed_codes'])) else False)
    halfyear_codes['main_2'] = halfyear_codes['CODE'].map(lambda x: True if (list(set(x) & set(ed_logic['ed_proc'])) and 'POS-23' in x) else False)
    halfyear_codes = halfyear_codes.loc[halfyear_codes['main_1'] | halfyear_codes['main_2'], ['ENC_ID', 'CODE']]
    if halfyear_codes.shape[0] == 0: return 0
    
    halfyear_codes['exc_1'] = halfyear_codes['CODE'].map(lambda x: False if list(set(x) & set(ed_logic['ed_mental'])) else True)
    halfyear_codes['exc_2'] = halfyear_codes['CODE'].map(lambda x: False if list(set(x) & set(ed_logic['eletroconvulsize'])) else True)
    halfyear_codes['exc_3'] = halfyear_codes['CODE'].map(lambda x: False if list(set(x) & set(ed_logic['psychiatry'])) else True)
    halfyear_codes = halfyear_codes.loc[halfyear_codes['exc_1'] & halfyear_codes['exc_2'] & halfyear_codes['exc_3'], ['ENC_ID', 'CODE']]
    if halfyear_codes.shape[0] == 0: return 0
    else: return 1

def member_cost(memberID, payer='CD_HEALTHFIRST',server='CARABWDB03', date_start=None, date_end=None):
    """
    Retrieve one member's codes
    @param memberID: a int
    @param payer: the name of the payer table
    @param server: CARA server on which the payer is located ('CARABWDB03')
    @param date_start: YYYY-MM-DD
    @param date_end: YYYY-MM-DD

    Return
    --------
    List of strings

    Examples
    --------
    >>> from enid.fetch_db import member_cost
    >>> ed_codes(1120565, "CD_HEALTHFIRST", date_start='2017-01-01', date_end='2017-12-31')
    10949.370018363
    """
    # initialize
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER='+server+';').cursor()
    date_start, date_end = str(date_start), str(date_end)

    sql = """
        SELECT SUM(ec.ecc_BilledAmount)
        FROM ["""+payer+"""].[dbo].[tbEncounter] e
        INNER JOIN ["""+payer+"""].[dbo].[tbEncounterCost] ec ON e.enc_id = ec.enc_id
        WHERE mem_id = """+str(memberID)+""" AND enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    """

    sql = re.sub(r" AND (e\.)?[a-zA-Z\_]+ BETWEEN 'None' AND 'None'", "", sql)
    data = cursor.execute(sql).fetchall()
    if data: return data[0][0]
    else:    return None

def batch_member_codes(payer='CD_HEALTHFIRST',server='CARABWDB03',memberIDs=None,date_start=None,date_end=None,mem_date_start=None,mem_date_end=None,get_client_id=True):
    """
    Retrieve a list of members' codes
    @param payer: the name of the payer table
    @param server: CARA server on which the payer is located ('CARABWDB03')
    @param memberIDs: a int or list of memberIDs; if None, fetch all members under the payer
    @param date_start: YYYY-MM-DD
    @param date_end: YYYY-MM-DD
    @param mem_date_start: YYYY-MM-DD, starting date to filter memberIDs by dateInserted
    @param mem_date_end: YYYY-MM-DD, ending date to filter memberIDs by dateInserted
    @param get_client_id: whether return member client IDs

    Return
    --------
    List of tuples (mem_id, *mem_ClientMemberID, encounter_id, Code)

    Examples
    --------
    >>> from enid.fetch_db import batch_member_codes
    >>> batch_member_codes("CD_HEALTHFIRST", memberIDs=[1120565])
    [(1120565, '130008347', 'ICD9-4011'),
     (1120565, '130008347', 'CPT-73562'),
     ...
     (1120565, '130008347', 'CPT-92012'),
     (1120565, '130008347', 'ICD9-78659')]
    """
    
    # initialize
    if memberIDs and not isinstance(memberIDs, list): memberIDs = [memberIDs]
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER='+server+';', autocommit=True).cursor()
    
    date_start, date_end, mem_date_start, mem_date_end = str(date_start), str(date_end), str(mem_date_start), str(mem_date_end)

    if memberIDs:
        cursor.execute("CREATE TABLE #MemberList (mem_id INT)")
        cursor.execute('\n'.join(["INSERT INTO #MemberList VALUES ({})".format(str(member)) for member in memberIDs]))
        while cursor.nextset(): pass
        cursor.commit()
        
    # AND mem_id IN ("""+', '.join([str(mem) for mem in memberIDs])+""")
    sql = """
    SET NOCOUNT ON
    
    SELECT tbm.mem_id, mem_ClientMemberID
    INTO #Temp
    FROM """+payer+""".dbo.tbMember tbm
        INNER JOIN #MemberList m ON tbm.mem_id = m.mem_id
    WHERE mem_ClientMemberID IS NOT NULL AND dateInserted BETWEEN '"""+mem_date_start+"""' AND '"""+mem_date_end+"""'
    ORDER BY mem_id DESC
    
    SELECT e.mem_id AS MemberID, tp.mem_ClientMemberID as MemberClientID, enc_serviceDate AS ServiceDate, CASE icdVersionInd WHEN 9 THEN 'ICD9' WHEN 10 THEN 'ICD10' ELSE 'ICD9' END AS CodeType, ed.icd_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterDiagnosis ed ON e.enc_id = ed.enc_id
    				INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
    WHERE e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT e.mem_id AS MemberID, tp.mem_ClientMemberID as MemberClientID, enc_serviceDate AS ServiceDate, 'CPT' AS CodeType, eCPT.cpt_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterCPT eCPT ON e.enc_id = eCPT.enc_id
    				INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
    WHERE e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT e.mem_id AS MemberID, tp.mem_ClientMemberID as MemberClientID, enc_serviceDate AS ServiceDate, 'DRG' AS CodeType, eDRG.DRG_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterDRG eDRG ON e.enc_id = eDRG.enc_id
    				INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
    WHERE e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT e.mem_id AS MemberID, tp.mem_ClientMemberID as MemberClientID, enc_serviceDate AS ServiceDate, 'HCPCS' AS CodeType, eHCPCS.HCPCS_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterHCPCS eHCPCS ON e.enc_id = eHCPCS.enc_id
    				INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
    WHERE e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""' 
    UNION
    SELECT e.mem_id AS MemberID, tp.mem_ClientMemberID as MemberClientID, enc_serviceDate AS ServiceDate, CASE icdVersionInd WHEN 9 THEN 'ICD9Proc' WHEN 10 THEN 'ICD10Proc' ELSE 'ICD9Proc' END AS CodeType, eProc.icd_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterProcedure eProc ON e.enc_id = eProc.enc_id
    				INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
    WHERE e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT p.mem_id AS MemberID, tp.mem_ClientMemberID as MemberClientID, pha_ServiceDate AS ServiceDate, 'NDC9' AS CodeType, NDC.ndcl_NDC9Code AS Code
    FROM """+payer+""".dbo.tbPharmacy p WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbPharmacyNDC NDC ON p.pha_id = NDC.pha_id
    				INNER JOIN #Temp tp ON tp.mem_id = p.mem_id
    WHERE p.pha_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    """
    
    if not memberIDs: sql = re.sub("INNER JOIN \#MemberList m ON tbm\.mem\_id = m.mem\_id", "", sql)
    sql = re.sub(r"AND mem_id IN \(None\)", "", sql)
    sql = re.sub(r"AND dateInserted BETWEEN 'None' AND 'None'", "", sql)
    sql = re.sub(r"WHERE (e|p)\.[a-zA-Z\_]+ BETWEEN 'None' AND 'None'", "", sql)
    sql = re.sub(r"INNER JOIN "+payer+"\.dbo\.tbFile f WITH\(NOLOCK\) ON (e|p)\.Fil_Id = f\.Fil_id and F\.fil_StartDate <= 'None'", "", sql)

    data = cursor.execute(sql).fetchall()
    if get_client_id:
        data = pd.DataFrame(list(set([(i[0], i[1], i[2], i[3]+'-'+i[4]) for i in data])))
        data.columns = ['MEMBER_ID', 'MEMBER_CLIENT_ID', 'TIME', 'CODE']
    else:
        data = pd.DataFrame(list(set([(i[0], i[2], i[3]+'-'+i[4]) for i in data])))
        data.columns = ['MEMBER_ID', 'TIME', 'CODE']

    data = data.sort_values(['MEMBER_ID', 'TIME'])
    if get_client_id: data = data.groupby(['MEMBER_ID', 'MEMBER_CLIENT_ID'])['TIME', 'CODE'].agg(list).reset_index()
    else:             data = data.groupby(['MEMBER_ID'])['TIME', 'CODE'].agg(list).reset_index()
    data = data.set_index('MEMBER_ID').to_dict('index')
    
#    all_members   = dict(get_members(payer, server, mem_date_start, mem_date_end))
#    if memberIDs: all_members = {i: all_members[i] for i in memberIDs}
#    other_members = {i: {'MEMBER_CLIENT_ID':all_members[i], 'TIME':[], 'CODE':[]} for i in list(set(all_members)-set(data))}
#    data.update(other_members)
    return data

############################# PRIVATE FUNCTIONS ###############################

def _get_variables(i, vec):
    if i < len(vec.all_variables): return vec.all_variables[i]
    else: return '_NONE_'