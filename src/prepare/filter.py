"""
Module to filter features and samples
"""
from collections.abc import Sequence
from typing import Optional
import logging

import numpy as np
import pandas as pd

from ..constants import drug_cols
from ..util import get_nmissing, get_excluded_numbers

from warnings import simplefilter 
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

def drop_highly_missing_features(
    df: pd.DataFrame, 
    missing_thresh: float, 
    keep_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """Drop features with high level of missingness
    
    Args:
        keep_cols: list of feature names to keep regardless of high missingness
    """
    nmissing = get_nmissing(df)
    mask = nmissing['Missing (%)'] > missing_thresh
    exclude_cols = nmissing.index[mask].drop(keep_cols, errors='ignore').tolist()
    msg = f'Dropping the following {len(exclude_cols)} features for missingness over {missing_thresh}%: {exclude_cols}'
    logger.info(msg)
    return df.drop(columns=exclude_cols)


def drop_unused_drug_features(df: pd.DataFrame) -> pd.DataFrame:
    # use 0 as a placeholder for nans and inf
    assert not (df[drug_cols] == 0).any().any() # ensure none of the drug feature value equals to 0
    df[drug_cols] = df[drug_cols].fillna(0).replace(np.inf, 0)

    # remove drugs given less than 10 times
    mask = (df[drug_cols] != 0).sum() < 10
    exclude = mask.index[mask].tolist()
    logger.info(f'Removing the following features for drugs given less than 10 times: {exclude}')
    df = df.drop(columns=exclude)

    return df


def drop_samples_outside_study_date(
    df: pd.DataFrame, 
    start_date: str = '2014-01-01', 
    end_date: str = '2019-12-31'
) -> pd.DataFrame:
    mask = df['treatment_date'].between(start_date, end_date)
    get_excluded_numbers(df, mask, context=f' before {start_date} and after {end_date}')
    df = df[mask]
    return df


def drop_samples_with_no_targets(df: pd.DataFrame, targ_cols: Sequence[str], missing_val=None) -> pd.DataFrame:
    if missing_val is None: 
        mask = df[targ_cols].isnull()
    else:
        mask = df[targ_cols] == missing_val
    mask = ~mask.all(axis=1)
    get_excluded_numbers(df, mask, context=' with no targets')
    df = df[mask]
    return df


def indicate_immediate_events(
    df: pd.DataFrame, 
    targ_cols: Sequence[str], 
    date_cols: Sequence[str],
    replace_val: int = -1
) -> pd.DataFrame:
    """Indicate samples where target event occured immediately after
     
    Indicate separately for each target
    
    Args:
        replace_val: The value to replace the target to indicate the exclusion
    """
    n_events = []
    for targ_col, date_col in zip(targ_cols, date_cols):
        days_until_event = df[date_col] - df['treatment_date']
        immediate_mask = days_until_event < pd.Timedelta('2 days')
        occured_mask = df[targ_col] == 1
        mask = immediate_mask & occured_mask
        df.loc[mask, targ_col] = replace_val
        n_events.append(sum(mask))
    logger.info(f'About {min(n_events)}-{max(n_events)} sessions had a target event '
                f'(e.g. {targ_cols[0]}) in less than 2 days.')
    return df


def exclude_immediate_events(df: pd.DataFrame, date_cols: Sequence[str]) -> pd.DataFrame:
    """Exclude samples where any one of the target events occured immediately after"""
    mask = False
    for col in date_cols:
        days_until_event = df[col] - df['treatment_date']
        mask |= days_until_event < pd.Timedelta('2 days')
    get_excluded_numbers(df, ~mask, context=f' in which patient had a target event in less than 2 days.')
    df = df[~mask]
    return df