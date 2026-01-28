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
        train_data, valid_data = self.random_split(train_data, test_size=0.2)

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
    """Stateful preprocessor: fit on train, transform all splits.

    Operations (in order):
    1. Remove low-variance columns
    2. Clip outliers (percentile-based)
    3. Normalize (z-score)
    """

    def __init__(
        self,
        var_thresh: float = 0.01,
        clip_percentile: tuple[float, float] = (0.001, 0.999),
        clip_cols: list[str] | None = None,
        norm_cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
    ):
        # Config
        self.var_thresh = var_thresh
        self.clip_percentile = clip_percentile
        self.clip_cols = clip_cols if clip_cols is not None else DEFAULT_CLIP_COLS
        self.norm_cols = norm_cols if norm_cols is not None else DEFAULT_NORM_COLS
        self.exclude_cols = exclude_cols if exclude_cols is not None else []

        # Fitted state
        self._low_var_cols: list[str] = []
        self._high_corr_cols: list[str] = []
        self._clip_bounds: dict[str, tuple[float, float]] = {}
        self._norm_params: dict[str, tuple[float, float]] = {}
        self._is_fitted: bool = False

    def fit(self, df: pl.DataFrame) -> "Preparer":
        """Learn transformation parameters from training data."""
        # 1. Identify low-variance columns
        self._low_var_cols = self._find_low_var_cols(df)

        # 2. Compute clip bounds for specified columns
        clip_cols = [
            c for c in self.clip_cols if c in df.columns and c not in self.exclude_cols
        ]
        for col in clip_cols:
            lower = df[col].quantile(self.clip_percentile[0])
            upper = df[col].quantile(self.clip_percentile[1])
            if lower is not None and upper is not None:
                self._clip_bounds[col] = (lower, upper)

        # 3. Compute normalization params for specified columns
        norm_cols = [
            c for c in self.norm_cols if c in df.columns and c not in self.exclude_cols
        ]
        for col in norm_cols:
            mean = df[col].mean()
            std = df[col].std()
            if mean is not None and std is not None and std > 0:
                self._norm_params[col] = (mean, std)

        self._is_fitted = True
        logger.info(
            f"Preparer fitted: dropping {len(self._low_var_cols)} low-variance cols, "
            f"clipping {len(self._clip_bounds)} cols, normalizing {len(self._norm_params)} cols"
        )
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply learned transformations to data."""
        if not self._is_fitted:
            raise RuntimeError("Preparer must be fitted before transform")

        # 1. Drop low-variance columns
        cols_to_drop = [
            c
            for c in self._low_var_cols
            if c in df.columns and c not in self.exclude_cols
        ]
        df = df.drop(cols_to_drop)

        # 2. Clip outliers
        clip_exprs = []
        for col, (lower, upper) in self._clip_bounds.items():
            if col in df.columns:
                clip_exprs.append(pl.col(col).clip(lower, upper))
        if clip_exprs:
            df = df.with_columns(clip_exprs)

        # 3. Normalize
        norm_exprs = []
        for col, (mean, std) in self._norm_params.items():
            if col in df.columns:
                norm_exprs.append(((pl.col(col) - mean) / std).alias(col))
        if norm_exprs:
            df = df.with_columns(norm_exprs)

        return df

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def _find_low_var_cols(self, df: pl.DataFrame) -> list[str]:
        """Find numeric columns with variance below threshold."""
        numeric_cols = df.select(pl.selectors.numeric()).columns
        numeric_cols = [col for col in numeric_cols if col not in self.exclude_cols]

        low_var = []
        for col in numeric_cols:
            var = df[col].var()
            if var is not None and var < self.var_thresh:
                low_var.append(col)
        return low_var


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


def build_features(
    df: pl.DataFrame,
    encode_cols: list[str] = None,
    impute_cols: list[str] = None,
    embed_cols: list[str] = None,
) -> pl.DataFrame:
    if encode_cols is None:
        encode_cols = DEFAULT_ENCODE_COLS
    if impute_cols is None:
        impute_cols = DEFAULT_IMPUTE_COLS
    if embed_cols is None:
        embed_cols = DEFAULT_EMBED_COLS

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
    df = df.to_dummies(columns=encode_cols)

    # map high-cardinal categories to indices for learned embeddings
    for col in embed_cols:
        df = df.with_columns(
            (pl.col(col).rank("dense") - 1).cast(pl.UInt32).alias(f"{col}_idx")
        )

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
    cols = [col for col in impute_cols if col in df.columns]
    df = df.with_columns(
        [
            # create missingness indicators for select columns
            *[
                pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_missing")
                for col in cols
            ],
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


def split_columns(
    df: pl.DataFrame,
    embed_cols: list[str],
    meta_cols: list[str],
    targ_cols: list[str],
) -> dict[str, pl.DataFrame]:
    """Split dataframe columns into logical groups.

    Args:
        df: Input dataframe
        meta_cols: Columns for metadata (not used in model)
        targ_cols: Columns for prediction targets
        embed_cols: Columns for text embedding features

    Returns:
        Dictionary with keys: "X_tabular", "X_embedding", "y", "meta"
    """
    # Filter to columns that exist
    meta_cols = [c for c in meta_cols if c in df.columns]
    target_cols = [c for c in targ_cols if c in df.columns]
    embedding_cols = [c for c in embed_cols if c in df.columns]

    # Tabular features = everything else
    exclude = set(meta_cols + target_cols + embedding_cols)
    tabular_cols = [c for c in df.columns if c not in exclude]

    return {
        "X_tabular": df.select(tabular_cols),
        "X_embedding": df.select(embedding_cols),
        "y": df.select(target_cols),
        "meta": df.select(meta_cols),
    }
