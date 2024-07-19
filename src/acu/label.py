"""
Module to extract acute care use labels
"""

from typing import Optional

from functools import partial

from tqdm import tqdm
import pandas as pd

from ml_common.util import split_and_parallelize


def get_event_labels(
    chemo_df: pd.DataFrame,
    event_df: pd.DataFrame,
    event_name: str,
    lookahead_window: int = 30,
    extra_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Extract labels for events (i.e. Emergency Department visit, hospitalization, etc) occuring within the next X days
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
    result = split_and_parallelize((chemo_df, event_df), worker)
    cols = ["index", f"target_{event_name}_date"] + [
        f"target_{col}" for col in extra_cols
    ]
    result = pd.DataFrame(result, columns=cols).set_index("index")
    chemo_df = chemo_df.join(result)

    # convert to binary label
    chemo_df[f"target_{event_name}"] = chemo_df[f"target_{event_name}_date"].notnull()

    return chemo_df


def event_worker(
    partition,
    event_name,
    lookahead_window: int = 30,
    extra_cols: Optional[list[str]] = None,
) -> list:
    if extra_cols is None:
        extra_cols = []

    chemo_df, event_df = partition
    result = []
    for mrn, chemo_group in tqdm(chemo_df.groupby("mrn")):
        event_group = event_df.query("mrn == @mrn")
        adm_dates = event_group["event_date"]

        for chemo_idx, visit_date in chemo_group["treatment_date"].items():
            # get target - closest event from visit date within lookahead window
            # NOTE: if event occured on treatment date, most likely the event
            # occured right after patient received treatment. We will deal with
            # it in downstream pipeline
            mask = adm_dates.between(
                visit_date, visit_date + pd.Timedelta(days=lookahead_window)
            )
            if not mask.any():
                continue

            # assert(sum(arrival_dates == arrival_dates[mask].min()) == 1)
            tmp = event_group.loc[mask].iloc[0]
            event_date = tmp["event_date"]
            extra_info = tmp[extra_cols].tolist()
            result.append([chemo_idx, event_date] + extra_info)
    return result
