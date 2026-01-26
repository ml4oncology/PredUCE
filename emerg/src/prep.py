import logging

import numpy as np
import polars as pl
from make_clinical_dataset.epic.combine import merge_closest_measurements
from make_clinical_dataset.shared.constants import (
    CANCER_DRUGS,
    LAB_COLS,
    SYMP_COLS,
)
from preduce.emerg.config import EMBEDDING_SECTIONS
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
class Preparer:
    """Prepare the data for model training"""

    def __init__(self):
        pass


###############################################################################
# Pipeline
###############################################################################
def load_data(
    tab_path: str,
    note_path: str,
    text_path: str,
    lookback_window: tuple[int, int] = None,
) -> pl.DataFrame:
    if lookback_window is None:
        lookback_window = (-30, -1)

    # Load data
    tab_df = pl.read_parquet(tab_path)
    note_df = pl.read_parquet(note_path, columns=["mrn", "clinic_date", "note_id"])
    text_df = pl.read_parquet(
        text_path,
        columns=["note_id"] + [f"{section}_text_id" for section in EMBEDDING_SECTIONS],
    )

    # Get the closest clinical note prior to assessment date within the lookback window
    note_df = note_df.rename(
        {"clinic_date": "prev_clinic_date", "note_id": "prev_note_id"}
    )
    df = merge_closest_measurements(
        tab_df,
        note_df,
        "assessment_date",
        "prev_clinic_date",
        merge_individually=False,
        time_window=lookback_window,
    )
    del tab_df, note_df

    # Include the text section ids
    df = df.join(text_df, left_on="prev_note_id", right_on="note_id", how="left")

    return df


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    # TODO: make it robust to missing columns
    # keep only the first treatment of a given week
    df = df.group_by_dynamic("assessment_date", every="7d", group_by="mrn").agg(
        pl.all().first()
    )

    # create an indicator on whether this is patient's very first treatment
    # TODO: move this to make-clinical-dataset
    df = df.with_columns(
        pl.col("days_since_last_treatment").is_null().alias("no_prior_treatment"),
    )

    # one-hot-encode categorical columns with low-cardinality
    # WARNING: assumes categories will remain constant over time
    df = df.to_dummies(columns=DEFAULT_ENCODE_COLS)

    # clip columns hueristically before imputation
    df = df.with_columns(
        [
            pl.col("days_since_starting_treatment").clip(lower_bound=-1),
            pl.col("prev_hospitalization_length_of_stay").clip(lower_bound=1),
        ]
    )

    # impute columns heuristrically (i.e. fill with zero)
    df = df.with_columns(
        [
            *[pl.col(col).fill_null(0) for col in CANCER_DRUGS],
            pl.col("radiation_dose_given").fill_null(0),
            pl.col("prev_hospitalization_length_of_stay").fill_null(0),
        ]
    )

    # impute columns via missing indicator approach (MIA)
    # use only the columns that exist in the data
    cols = [col for col in DEFAULT_IMPUTE_COLS if col in df.columns]
    df = df.with_columns(
        [
            # create missingness indicators for select columns
            *[pl.col(col).is_null().alias(f"{col}_missing") for col in cols],
            # fill missing values with -1
            # NOTE: the model will learn from the appropriate indicator to ignore this value
            #   i.e. ignore  "days_since_prev_ED_visit" when "num_prior_ED_visits_within_5_years" == 0
            #   i.e. ignore  "days_since_last_treatment" when "no_prior_treatment" == 1
            #   i.e. ignore "hemoglobin" when "hemoglobin_missing" == 1
            *[pl.col(col).fill_null(-1) for col in cols],
        ]
    )
    # drop unnecessary missingness indicators
    df = df.drop(
        "days_since_last_treatment_missing",
        "days_since_prev_ED_visit_missing",
        "days_since_prev_hospitalization_missing",
        strict=False,
    )

    return df
