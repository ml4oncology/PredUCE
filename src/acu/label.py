"""
Module to extract acute care use labels
"""

from typing import Optional

from functools import partial

from tqdm import tqdm
import pandas as pd

from ml_common.util import split_and_parallelize


def get_event_labels(
    df: pd.DataFrame,
    event_df: pd.DataFrame,
    event_name: str,
    lookahead_window: int = 30,
    extra_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Extract labels for events (i.e. emergency department visit, hospitalization, etc) occuring within the next X days
    after visit date

    Args:
        event_df: The processed event data from https://github.com/ml4oncology/make-clinical-dataset
        lookahead_window: The lookahead window in terms of days after visit date in which labels can be extracted
        extra_cols: Additional label information to extract (e.g. triage category)
    """
    if extra_cols is None:
        extra_cols = []

    # extract the future event dates
    worker = partial(
        event_worker,
        event_name=event_name,
        lookahead_window=lookahead_window,
        extra_cols=extra_cols,
    )
    result = split_and_parallelize((df, event_df), worker)
    cols = ["index", f"target_{event_name}_date"] + [
        f"target_{col}" for col in extra_cols
    ]
    result = pd.DataFrame(result, columns=cols).set_index("index")
    df = df.join(result)

    # convert to binary label
    df[f"target_{event_name}"] = df[f"target_{event_name}_date"].notnull()

    return df


def event_worker(
    partition,
    event_name,
    lookahead_window: int = 30,
    extra_cols: Optional[list[str]] = None,
) -> list:
    if extra_cols is None:
        extra_cols = []

    df, event_df = partition
    result = []
    for mrn, group in df.groupby("mrn"):
        event_group = event_df.query("mrn == @mrn")
        adm_dates = event_group["event_date"]

        for idx, visit_date in group["assessment_date"].items():
            # get the closest event from visit date within lookahead window
            mask = adm_dates.between(
                visit_date, visit_date + pd.Timedelta(days=lookahead_window)
            )
            if not mask.any():
                continue

            # assert(sum(arrival_dates == arrival_dates[mask].min()) == 1)
            tmp = event_group.loc[mask].iloc[0]
            event_date = tmp["event_date"]
            extra_info = tmp[extra_cols].tolist()
            result.append([idx, event_date] + extra_info)
    return result
