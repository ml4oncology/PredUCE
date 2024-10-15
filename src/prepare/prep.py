"""
Module to prepare data for model consumption
"""

from typing import Optional

import os
import subprocess

import pandas as pd
import yaml


def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing data that can be filled heuristically"""
    # fill the following missing data with 0
    for col in ["num_prior_ED_visits_within_5_years", "days_since_starting_treatment"]:
        df[col] = df[col].fillna(0)

    # fill the following missing data with the maximum value
    for col in ["days_since_last_treatment", "days_since_prev_ED_visit"]:
        df[col] = df[col].fillna(df[col].max())

    return df
