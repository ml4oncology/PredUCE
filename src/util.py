import itertools
import logging
import multiprocessing as mp
import os
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

###############################################################################
# I/O
###############################################################################
def load_pickle(save_dir: str, filename: str, err_msg=None):
    filepath = f'{save_dir}/{filename}.pkl'
    with open(filepath, 'rb') as file:
        output = pickle.load(file)
    return output


def save_pickle(result, save_dir: str, filename: str):
    filepath = f'{save_dir}/{filename}.pkl'
    with open(filepath, 'wb') as file:    
        pickle.dump(result, file)


def initialize_folders():
    main_folders = ['logs', 'models', 'result/tables']
    for folder in main_folders:
        if not os.path.exists(f'./{folder}'):
            os.makedirs(f'./{folder}')

###############################################################################
# Multiprocessing
###############################################################################
def parallelize(generator, worker, processes: int = 4) -> list:
    pool = mp.Pool(processes=processes)
    result = pool.map(worker, generator)
    pool.close()
    pool.join() # wait for all threads
    result = list(itertools.chain(*result))
    return result

def split_and_parallelize(data, worker, split_by_mrns: bool = True, processes: int = 4) -> list:
    """Split up the data and parallelize processing of data
    
    Args:
        data: Supports a sequence, pd.DataFrame, or tuple of pd.DataFrames 
            sharing the same patient ids
        split_by_mrns: If True, split up the data by patient ids
    """
    generator = []
    if split_by_mrns:
        mrns = data[0]['mrn'] if isinstance(data, tuple) else data['mrn']
        mrn_groupings = np.array_split(mrns.unique(), processes)
        if isinstance(data, tuple):
            for mrn_grouping in mrn_groupings:
                items = tuple(df[df['mrn'].isin(mrn_grouping)] for df in data)
                generator.append(items)
        else:
            for mrn_grouping in mrn_groupings:
                item = data[mrns.isin(mrn_grouping)]
                generator.append(item)
    else:
        # splits df into x number of partitions, where x is number of processes
        generator = np.array_split(data, processes)
    return parallelize(generator, worker, processes=processes)

###############################################################################
# Data Descriptions
###############################################################################
def get_nunique_categories(df: pd.DataFrame) -> pd.DataFrame:
    catcols = df.dtypes[df.dtypes == object].index.tolist()
    return pd.DataFrame(
        df[catcols].nunique(), columns=['Number of Unique Categories']
    ).T

def get_nmissing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum() # number of nans for each column
    missing = missing[missing != 0] # remove columns without missing values
    missing = pd.DataFrame(missing, columns=['Missing (N)'])
    missing['Missing (%)'] = (missing['Missing (N)'] / len(df) * 100).round(3)
    return missing.sort_values(by='Missing (N)')

def get_excluded_numbers(df, mask: pd.Series, context: str = '.') -> None:
    """Report the number of patients and sessions that were excluded"""
    N_sessions = sum(~mask)
    N_patients = len(set(df['mrn']) - set(df.loc[mask, 'mrn']))
    logger.info(f'Removing {N_patients} patients and {N_sessions} sessions{context}')