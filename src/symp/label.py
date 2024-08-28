"""
Module to extract symptom deterioration labels
"""

from typing import Optional

from functools import partial

from tqdm import tqdm
import pandas as pd

from ml_common.constants import SYMP_COLS
from ml_common.util import split_and_parallelize


def convert_to_binary_symptom_labels(
    df: pd.DataFrame, scoring_map: Optional[dict[str, int]] = None
) -> pd.DataFrame:
    """Convert label to 1 (positive), 0 (negative), or -1 (missing/exclude)

    Label is positive if symptom deteriorates (score increases) by X points
    """
    if scoring_map is None:
        scoring_map = {col: 3 for col in SYMP_COLS}
    for base_col, pt in scoring_map.items():
        continuous_targ_col = f"target_{base_col}_change"
        discrete_targ_col = f"target_{base_col}_{pt}pt_change"
        missing_mask = df[continuous_targ_col].isnull()
        df[discrete_targ_col] = (df[continuous_targ_col] >= pt).astype(int)
        df.loc[missing_mask, discrete_targ_col] = -1

        # If baseline score is alrady high, we exclude them
        df.loc[df[base_col] > 10 - pt, discrete_targ_col] = -1
    return df


def get_symptom_labels(
    chemo_df: pd.DataFrame, symp_df: pd.DataFrame, lookahead_window: int = 30
) -> pd.DataFrame:
    """Extract labels for symptom deterioration within the next X days after visit date

    Args:
        symp: The processed symptom data from https://github.com/ml4oncology/make-clinical-dataset
        lookahead_window: The lookahead window in terms of days after visit date in which labels can be extracted
    """
    # extract the target symptom scores
    worker = partial(symptom_worker, lookahead_window=lookahead_window)
    result = split_and_parallelize((chemo_df, symp_df), worker)
    cols = []
    for symp in SYMP_COLS:
        cols += [f"target_{symp}_survey_date", f"target_{symp}"]
    result = pd.DataFrame(result, columns=["index"] + cols).set_index("index")
    chemo_df = pd.concat([chemo_df, result], axis=1)

    # compute target symptom score change
    for symp in SYMP_COLS:
        chemo_df[f"target_{symp}_change"] = chemo_df[f"target_{symp}"] - chemo_df[symp]

    return chemo_df


def symptom_worker(partition, lookahead_window: int = 30) -> list:
    chemo_df, symp_df = partition
    result = []
    for mrn, chemo_group in tqdm(
        chemo_df.groupby("mrn"), desc="Getting symptom labels..."
    ):
        symp_group = symp_df.query("mrn == @mrn")
        surv_dates = symp_group["survey_date"]

        for chemo_idx, visit_date in chemo_group["treatment_date"].items():
            # NOTE: the baseline ESAS score can include surveys taken on visit date.
            # To make sure the target ESAS score does not overlap with baseline ESAS score,
            # only take surveys AFTER the visit date
            mask = surv_dates.between(
                visit_date
                + pd.Timedelta(days=1),  # NOTE: you can also just do inclusive='right'
                visit_date + pd.Timedelta(days=lookahead_window),
            )
            if not mask.any():
                continue

            data = []
            for symp in SYMP_COLS:
                # take the max (worst) symptom scores within the target timeframe
                scores = symp_group.loc[mask, symp]
                idx = None if all(scores.isnull()) else scores.idxmax(skipna=True)
                data += (
                    [None, None]
                    if idx is None
                    else [surv_dates[idx], symp_group.loc[idx, symp]]
                )
            result.append([chemo_idx] + data)
    return result
