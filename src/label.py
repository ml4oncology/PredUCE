"""
Module to extract and view labels
"""
from collections import defaultdict
from collections.abc import Sequence
from functools import partial

from tqdm import tqdm
import numpy as np
import pandas as pd

from .constants import symp_cols
from .util import get_excluded_numbers, split_and_parallelize

###############################################################################
# Symptom
###############################################################################
def convert_to_binary_symptom_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert label to 1 (positive), 0 (negative), or -1 (missing/exclude)

    Label is positive if symptom deteriorates (score increases) by X points
    """
    score_increase_map = {
        'esas_pain': 4,
        'esas_tiredness': 4,
        'esas_nausea': 4,
        'esas_depression': 4,
        'esas_anxiety': 4,
        'esas_drowsiness': 4,
        'esas_appetite': 4,
        'esas_well_being': 4,
        'esas_shortness_of_breath': 4,
        'patient_ecog': 1,
    }
    for base_col in symp_cols:
        targ_col = f'target_{base_col}_change'
        missing_mask = df[targ_col].isnull()
        df[targ_col] = (df[targ_col] >= score_increase_map[base_col]).astype(int)
        df.loc[missing_mask, targ_col] = -1

        # If baseline score is alrady 7+, we exclude them
        df.loc[df[base_col] > 6, targ_col] = -1

        # If not a positive label example, assign a NaT to the survey date. 
        # Useful for later, when removing immediate events
        df.loc[df[targ_col] != 1, targ_col.replace('change', 'survey_date')] = pd.NaT
    return df


def get_symptom_labels(chemo_df: pd.DataFrame, symp_df: pd.DataFrame, lookahead_window: int = 30) -> pd.DataFrame:
    """Extract labels for symptom deterioration within the next X days after visit date

    Args:
        symp: The processed symptom data from https://github.com/ml4oncology/make-clinical-dataset
        lookahead_window: The lookahead window after visit date in which labels can be extracted
    """
    # remove the following symptoms (too little data)
    symp_df = symp_df.drop(columns=['esas_constipation', 'esas_vomiting', 'esas_diarrhea'])

    # extract the target symptom scores
    worker = partial(symptom_worker, lookahead_window=lookahead_window)
    result = split_and_parallelize((chemo_df, symp_df), worker)
    cols = []
    for col in symp_cols:
        cols += [f'target_{col}_survey_date', f'target_{col}']
    result = pd.DataFrame(result, columns=['index']+cols).set_index('index')
    chemo_df = pd.concat([chemo_df, result], axis=1)
    
    # compute target symptom score change
    for col in symp_cols: 
        chemo_df[f'target_{col}_change'] = chemo_df[f'target_{col}'] - chemo_df[col]

    return chemo_df


def symptom_worker(partition, lookahead_window: int = 30) -> list:
    chemo_df, symp_df = partition
    symp_cols = symp_df.columns.drop(['mrn', 'survey_date'])
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
            for col in symp_cols:
                # take the max (worst) symptom scores within the target timeframe
                idx = symp_group.loc[mask, col].idxmax()
                data += [None, None] if np.isnan(idx) else [surv_dates[idx], symp_group.loc[idx, col]]
            result.append([chemo_idx]+data)
    return result


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
        dists = {split: group.apply(pd.value_counts) 
                 for split, group in Y.groupby(metainfo['split'])}
    dists['Total'] = dists['Train'] + dists['Valid'] + dists['Test']
    dists = pd.concat(dists).T
    return dists


###############################################################################
# Acute Care Use
###############################################################################
def get_event_labels(chemo_df: pd.DataFrame, event_df: pd.DataFrame, lookahead_window: int = 30) -> pd.DataFrame:
    """Extract labels for events (i.e. Emergency Department visit, hospitalization, etc) occuring within the next X days
    after visit date

    Args:
        event_df: The processed event data from https://github.com/ml4oncology/make-clinical-dataset
        lookahead_window: The lookahead window after visit date in which labels can be extracted
    """
    # extract the future event dates
    worker = partial(event_worker, lookahead_window=lookahead_window)
    result = split_and_parallelize((chemo_df, event_df), worker)
    cols = ['index', 'target_ED_visit_date', 'target_triage_category']
    result = pd.DataFrame(result, columns=cols).set_index('index')
    chemo_df = pd.concat([chemo_df, result], axis=1)
    
    # compute target symptom score change
    for col in symp_cols: 
        chemo_df[f'target_{col}_change'] = chemo_df[f'target_{col}'] - chemo_df[col]

    return chemo_df


def emergency_department_worker(partition) -> list:
    chemo_df, event_df = partition
    result = []
    for mrn, chemo_group in tqdm(chemo_df.groupby('mrn')):
        event_group = event_df.query('mrn == @mrn')
        adm_dates = event_group['adm_date']
        
        for chemo_idx, visit_date in chemo_group['treatment_date'].items():
            # get target - closest event from visit date
            # NOTE: if event occured on treatment date, most likely the event 
            # occured right after patient received treatment. We will deal with
            # it in downstream pipeline
            mask = adm_dates >= visit_date
            if not mask.any():
                continue

            # assert(sum(arrival_dates == arrival_dates[mask].min()) == 1)
            tmp = event_group.loc[mask].iloc[0]
            adm_date = tmp['adm_date']
            triage_category = tmp['triage_category']
            result.append([chemo_idx, adm_date, triage_category])
    return result
