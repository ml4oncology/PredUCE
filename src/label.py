"""
Module to extract and view labels
"""
from typing import Optional

from collections import defaultdict
from collections.abc import Sequence
from functools import partial

from tqdm import tqdm
import numpy as np
import pandas as pd

from .constants import symp_cols
from .util import get_excluded_numbers, split_and_parallelize


def get_label_distribution(
    Y: pd.DataFrame, 
    metainfo: pd.DataFrame, 
    with_respect_to: str = 'sessions'
) -> pd.DataFrame:
    if with_respect_to == 'patients':
        dists = {}
        for split, group in Y.groupby(metainfo['split']):
            count = defaultdict(dict)
            mrn = metainfo.loc[group.index, 'mrn']
            for target, labels in group.items():
                count[1][target] = mrn[labels == 1].nunique()
                count[0][target] = mrn.nunique() - count[1][target]
            dists[split] = pd.DataFrame(count).T
    elif with_respect_to == 'sessions':
        dists = {split: group.apply(lambda x: x.value_counts()) 
                 for split, group in Y.groupby(metainfo['split'])}
    dists['Total'] = dists['Train'] + dists['Valid'] + dists['Test']
    dists = pd.concat(dists).T
    return dists


###############################################################################
# Symptom
###############################################################################
def convert_to_binary_symptom_labels(df: pd.DataFrame, scoring_map: Optional[dict[str, int]] = None) -> pd.DataFrame:
    """Convert label to 1 (positive), 0 (negative), or -1 (missing/exclude)

    Label is positive if symptom deteriorates (score increases) by X points
    """
    if scoring_map is None: scoring_map = {col: 3 for col in symp_cols}
    for base_col, pt in scoring_map.items():
        continuous_targ_col = f'target_{base_col}_change'
        discrete_targ_col = f'target_{base_col}_{pt}pt_change'
        missing_mask = df[continuous_targ_col].isnull()
        df[discrete_targ_col ] = (df[continuous_targ_col] >= pt).astype(int)
        df.loc[missing_mask, discrete_targ_col] = -1

        # If baseline score is alrady high, we exclude them
        df.loc[df[base_col] > 10 - pt, discrete_targ_col] = -1
    return df


def get_symptom_labels(chemo_df: pd.DataFrame, symp_df: pd.DataFrame, lookahead_window: int = 30) -> pd.DataFrame:
    """Extract labels for symptom deterioration within the next X days after visit date

    Args:
        symp: The processed symptom data from https://github.com/ml4oncology/make-clinical-dataset
        lookahead_window: The lookahead window in terms of days after visit date in which labels can be extracted
    """
    # extract the target symptom scores
    worker = partial(symptom_worker, lookahead_window=lookahead_window)
    result = split_and_parallelize((chemo_df, symp_df), worker)
    cols = []
    for symp in symp_cols:
        cols += [f'target_{symp}_survey_date', f'target_{symp}']
    result = pd.DataFrame(result, columns=['index']+cols).set_index('index')
    chemo_df = pd.concat([chemo_df, result], axis=1)
    
    # compute target symptom score change
    for symp in symp_cols: 
        chemo_df[f'target_{symp}_change'] = chemo_df[f'target_{symp}'] - chemo_df[symp]

    return chemo_df


def symptom_worker(partition, lookahead_window: int = 30) -> list:
    chemo_df, symp_df = partition
    result = []
    for mrn, chemo_group in tqdm(chemo_df.groupby('mrn')):
        symp_group = symp_df.query('mrn == @mrn')
        surv_dates = symp_group['survey_date']

        for chemo_idx, visit_date in chemo_group['treatment_date'].items():
            # NOTE: the baseline ESAS score can include surveys taken on visit date.
            # To make sure the target ESAS score does not overlap with baseline ESAS score,
            # only take surveys AFTER the visit date
            mask = surv_dates.between(
                visit_date + pd.Timedelta(days=1), # NOTE: you can also just do inclusive='right'
                visit_date + pd.Timedelta(days=lookahead_window)
            )
            if not mask.any():
                continue

            data = []
            for symp in symp_cols:
                # take the max (worst) symptom scores within the target timeframe
                idx = symp_group.loc[mask, symp].idxmax()
                data += [None, None] if np.isnan(idx) else [surv_dates[idx], symp_group.loc[idx, symp]]
            result.append([chemo_idx]+data)
    return result


###############################################################################
# Acute Care Use
###############################################################################
def get_event_labels(
    chemo_df: pd.DataFrame, 
    event_df: pd.DataFrame, 
    event_name: str,
    lookahead_window: int = 30,
    extra_cols: Optional[list[str]] = None
) -> pd.DataFrame:
    """Extract labels for events (i.e. Emergency Department visit, hospitalization, etc) occuring within the next X days
    after visit date

    Args:
        event_df: The processed event data from https://github.com/ml4oncology/make-clinical-dataset
        lookahead_window: The lookahead window in terms of days after visit date in which labels can be extracted
        extra_cols: Additional label information to extract (e.g. triage category)
    """
    if extra_cols is None: extra_cols = []

    # extract the future event dates
    worker = partial(event_worker, event_name=event_name, lookahead_window=lookahead_window, extra_cols=extra_cols)
    result = split_and_parallelize((chemo_df, event_df), worker)
    cols = ['index', f'target_{event_name}_date'] + [f'target_{col}' for col in extra_cols]
    result = pd.DataFrame(result, columns=cols).set_index('index')
    chemo_df = chemo_df.join(result)
    
    # convert to binary label
    chemo_df[f'target_{event_name}'] = chemo_df[f'target_{event_name}_date'].notnull()

    return chemo_df


def event_worker(partition, event_name, lookahead_window: int = 30, extra_cols: Optional[list[str]] = None) -> list:
    if extra_cols is None: extra_cols = []
    
    chemo_df, event_df = partition
    result = []
    for mrn, chemo_group in tqdm(chemo_df.groupby('mrn')):
        event_group = event_df.query('mrn == @mrn')
        adm_dates = event_group['event_date']
        
        for chemo_idx, visit_date in chemo_group['treatment_date'].items():
            # get target - closest event from visit date within lookahead window
            # NOTE: if event occured on treatment date, most likely the event 
            # occured right after patient received treatment. We will deal with
            # it in downstream pipeline
            mask = adm_dates.between(visit_date, visit_date + pd.Timedelta(days=lookahead_window))
            if not mask.any():
                continue

            # assert(sum(arrival_dates == arrival_dates[mask].min()) == 1)
            tmp = event_group.loc[mask].iloc[0]
            event_date = tmp['event_date']
            extra_info = tmp[extra_cols].tolist()
            result.append([chemo_idx, event_date] + extra_info)
    return result
