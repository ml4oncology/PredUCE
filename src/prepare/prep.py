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
    col = "num_prior_ED_visits_within_5_years"
    df[col] = df[col].fillna(0)

    # fill the following missing data with the maximum value
    for col in ["days_since_last_treatment", "days_since_prev_ED_visit"]:
        df[col] = df[col].fillna(df[col].max())

    return df


###############################################################################
# Anchor
###############################################################################
def anchor_features_to_assessment_dates(
    script_path: str, output_filename: Optional[str] = None
):
    """Create feature dataset anchored to assessment dates by
    calling the make-clincial-dataset package's combine_features script

    See https://github.com/ml4oncology/make-clinical-dataset

    Args:
        script_path: path to the combine_features script
    """
    if output_filename is None:
        output_filename = "assessment_centered_feature_dataset"

    cfg = dict(
        trt_lookback_window=[-28, -1],
        lab_lookback_window=[-7, -1],
        symp_lookback_window=[-30, -1],
        ed_visit_lookback_window=5,
    )
    with open("./data/config.yaml", "w") as file:
        yaml.dump(cfg, file)

    # ensure the necessary files are present to run the script
    # you will need to copy the following from make-clinical-dataset/data/interim to ./data/interim
    for filename in [
        "lab",
        "symptom",
        "demographic",
        "emergency_room_visit",
        "treatment",
    ]:
        assert os.path.exists(
            f"./data/interim/{filename}.parquet.gzip"
        ), "Please retreive the necessary files"
    # same with this file from make-clinical-dataset/data/external to ./data/external
    assert os.path.exists(
        "./data/external/opis_drug_list.csv"
    ), "Please retreive the necessary files"

    # call the script
    res = subprocess.run(
        [
            "python",
            f"{script_path}/combine_features.py",
            "--align-on",
            "./data/processed/assessment_dates.csv",
            "--date-column",
            "assessment_date",
            "--data-dir",
            "./data",
            "--output-dir",
            "./data/processed",
            "--output-filename",
            output_filename,
            "--config-path",
            "./data/config.yaml",
        ],
        capture_output=True,
    )
    assert res.returncode == 0
