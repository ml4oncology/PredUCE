"""
Module to split the data
"""

import logging

from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

from .filter import get_excluded_numbers

logger = logging.getLogger(__name__)


# Data splitting
def create_train_val_test_splits(
    data: pd.DataFrame, split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the training, validation, and testing set"""
    # split data temporally based on patients first visit date
    train_data, test_data = create_temporal_cohort(data, split_date)
    # create validation set from train data (80-20 split)
    train_data, valid_data = create_random_split(train_data, test_size=0.2)
    return train_data, valid_data, test_data


def create_temporal_cohort(
    df: pd.DataFrame, split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the development and testing cohort by partitioning on split_date"""
    first_date = df.groupby("mrn")["treatment_date"].min()
    first_date = df["mrn"].map(first_date)
    mask = first_date <= split_date
    dev_cohort, test_cohort = df[mask].copy(), df[~mask].copy()

    # remove visits in the dev_cohort that occured after split_date
    mask = dev_cohort["treatment_date"] <= split_date
    get_excluded_numbers(
        dev_cohort, mask, f" that occured after {split_date} in the development cohort"
    )
    dev_cohort = dev_cohort[mask]

    disp = lambda x: f"NSessions={len(x)}. NPatients={x.mrn.nunique()}"
    msg = f"Development Cohort: {disp(dev_cohort)}. Contains all patients whose first visit was on or before {split_date}"
    logger.info(msg)
    msg = f"Test Cohort: {disp(test_cohort)}. Contains all patients whose first visit was after {split_date}"
    logger.info(msg)

    return dev_cohort, test_cohort


def create_random_split(
    df: pd.DataFrame, test_size: float, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data radnomly based on patient ids"""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    patient_ids = df["mrn"]
    train_idxs, test_idxs = next(gss.split(df, groups=patient_ids))
    train_data = df.iloc[train_idxs].copy()
    test_data = df.iloc[test_idxs].copy()
    return train_data, test_data
