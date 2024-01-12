"""
Module to filter features and samples
"""
from collections.abc import Sequence

import pandas as pd

from .. import logger
from ..util import get_nmissing, get_excluded_numbers

from warnings import simplefilter 
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def drop_highly_missing_features(df: pd.DataFrame, missing_thresh: float) -> pd.DataFrame:
    """Drop features with high level of missingness"""
    nmissing = get_nmissing(df)
    mask = nmissing['Missing (%)'] > missing_thresh
    exclude_cols = nmissing.index[mask].tolist()
    msg = f'Dropping the following {len(exclude_cols)} features for missingness over {missing_thresh}%: {exclude_cols}'
    logger.info(msg)
    return df.drop(columns=exclude_cols)


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


def exclude_immediate_events(
    df: pd.DataFrame, 
    targ_cols: Sequence[str], 
    date_cols: Sequence[str]
) -> pd.DataFrame:
    """Exclude samples where target event occured immediately after
     
    Exclude separately for each target
    """
    for targ_col, date_col in zip(targ_cols, date_cols):
        days_until_event = df[date_col] - df['treatment_date']
        immediate_mask = days_until_event < pd.Timedelta('2 days')
        df.loc[immediate_mask, targ_col] = -1
    return df


def _exclude_immediate_events(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """ARCHIVED - If any one of the targets are immediate, we exclude them"""
    mask = False
    for col in cols:
        days_until_event = df[col] - df['treatment_date']
        mask |= days_until_event < pd.Timedelta('2 days')
    get_excluded_numbers(df, ~mask, context=f' in which patient had an event in less than 2 days.')
    df = df[~mask]
    return df