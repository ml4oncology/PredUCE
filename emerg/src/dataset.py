import logging

import numpy as np
import polars as pl
from make_clinical_dataset.shared.constants import (
    CANCER_DRUGS,
    LAB_COLS,
    SYMP_COLS,
)
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

###############################################################################
# Default columns for data transformations
###############################################################################
# imputation
DEFAULT_IMPUTE_COLS = (
    [
        "days_since_last_treatment",
        "days_since_prev_ED_visit",
        "days_since_prev_hospitalization",
        "body_surface_area",
        "height",
        "weight",
        "line_of_therapy",
    ]
    + LAB_COLS
    + SYMP_COLS
)

# one-hot encoding (low-cardinal categorical features)
DEFAULT_ENCODE_COLS = ["intent", "sex"]

# learned embedding (high-cardinal categorical features)
DEFAULT_EMBED_COLS = [
    "primary_site_desc",
]

# outlier clipping
DEFAULT_CLIP_COLS = ["body_surface_area", "height", "weight"] + LAB_COLS

# normalization
DEFAULT_NORM_COLS = (
    [
        "days_since_last_treatment",
        "days_since_prev_ED_visit",
        "days_since_prev_hospitalization",
        "days_since_starting_treatment",
        "num_prior_ED_visits_within_5_years",
        "age",
        "body_surface_area",
        "height",
        "weight",
        "cycle_number",
        "line_of_therapy",
    ]
    + CANCER_DRUGS
    + LAB_COLS
    + SYMP_COLS
)


###############################################################################
# Splitting
###############################################################################
class Splitter:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def split_data(
        self, df: pl.DataFrame, split_date: str, **kwargs
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Create the training, validation, and testing set"""
        # split data temporally based on patients first visit date
        train_data, test_data = self.temporal_split(df, split_date=split_date, **kwargs)

        # create validation set from train data (80-20 split)
        train_data, valid_data = self.random_split(
            train_data, test_size=0.2, random_state=self.random_state
        )

        return train_data, valid_data, test_data

    def temporal_split(
        self,
        df: pl.DataFrame,
        split_date: str = "2022-01-01",
        visit_col: str = "treatment_date",
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split the data temporally based on patient's first visit date"""
        first_date = pl.col(visit_col).min().over("mrn")
        mask = first_date <= pl.lit(split_date).str.to_date()
        dev_cohort, test_cohort = df.filter(mask), df.filter(~mask)
        return dev_cohort, test_cohort

    def random_split(
        self,
        df: pl.DataFrame,
        test_size: float,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split the data randomly based on patient id"""
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=self.random_state
        )
        groups = df["mrn"].to_numpy()
        indices = np.arange(len(df))
        train_idx, test_idx = next(gss.split(indices, groups=groups))
        return df[train_idx], df[test_idx]


###############################################################################
# Transformation
###############################################################################
class PrepData:
    """Prepare the data for model training"""

    def __init__(self):
        pass
