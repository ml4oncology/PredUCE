"""
Module to filter features and samples
"""

from collections.abc import Sequence
import logging

from tqdm import tqdm
import pandas as pd

from ml_common.util import get_excluded_numbers

from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


def drop_samples_outside_study_date(
    df: pd.DataFrame, start_date: str = "2014-01-01", end_date: str = "2019-12-31"
) -> pd.DataFrame:
    mask = df["treatment_date"].between(start_date, end_date)
    get_excluded_numbers(df, mask, context=f" before {start_date} and after {end_date}")
    df = df[mask]
    return df


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
        days_until_event = df[date_col] - df["treatment_date"]
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
        days_until_event = df[col] - df["treatment_date"]
        mask |= days_until_event < pd.Timedelta("2 days")
    get_excluded_numbers(
        df, ~mask, context=f" in which patient had a target event in less than 2 days."
    )
    df = df[~mask]
    return df


def keep_only_one_per_week(df: pd.DataFrame) -> list[int]:
    """Keep only the first treatment session of a given week
    Drop all other sessions
    """
    keep_idxs = []
    for mrn, group in tqdm(
        df.groupby("mrn"), desc="Getting the first sessions of a given week..."
    ):
        previous_date = pd.Timestamp.min
        for i, visit_date in group["treatment_date"].items():
            if visit_date >= previous_date + pd.Timedelta(days=7):
                keep_idxs.append(i)
                previous_date = visit_date
    get_excluded_numbers(
        df, mask=df.index.isin(keep_idxs), context=f" not first of a given week"
    )
    df = df.loc[keep_idxs]
    return df
