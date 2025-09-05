"""
Module to filter features and samples
"""

import logging
from collections.abc import Sequence
from warnings import simplefilter

import pandas as pd
from make_clinical_dataset.epr.util import get_excluded_numbers

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


def indicate_immediate_events(
    df: pd.DataFrame,
    targ_cols: Sequence[str],
    date_cols: Sequence[str],
    replace_val: int = -1,
) -> pd.DataFrame:
    """Indicate samples where target event occured immediately after

    Indicate separately for each target

    Args:
        replace_val: The value to replace the target to indicate the exclusion
    """
    n_events = []
    for targ_col, date_col in zip(targ_cols, date_cols):
        days_until_event = df[date_col] - df["assessment_date"]
        immediate_mask = days_until_event < pd.Timedelta("2 days")
        occured_mask = df[targ_col] == 1
        mask = immediate_mask & occured_mask
        df.loc[mask, targ_col] = replace_val
        n_events.append(sum(mask))
    logger.info(
        f"About {min(n_events)}-{max(n_events)} sessions had a target event "
        f"(e.g. {targ_cols[0]}) in less than 2 days."
    )
    return df


def exclude_immediate_events(
    df: pd.DataFrame, date_cols: Sequence[str]
) -> pd.DataFrame:
    """Exclude samples where any one of the target events occured immediately after"""
    mask = False
    for col in date_cols:
        days_until_event = df[col] - df["assessment_date"]
        mask |= days_until_event < pd.Timedelta("2 days")
    get_excluded_numbers(
        df, ~mask, context=" in which patient had a target event in less than 2 days."
    )
    df = df[~mask]
    return df
