"""
Module to prepare data for model consumption
"""

from typing import Optional
import logging

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

from .engineer import collapse_rare_categories
from ..constants import (
    drug_cols,
    lab_cols,
    lab_change_cols,
    symp_cols,
    symp_change_cols,
)

logger = logging.getLogger(__name__)


class Imputer:
    """Impute missing data by mean, mode, or median"""

    def __init__(self):
        self.impute_cols = {
            "mean": lab_cols.copy() + lab_change_cols.copy(),
            "most_frequent": symp_cols.copy()
            + symp_change_cols.copy()
            + ["patient_ecog", "patient_ecog_change"],
            "median": [],
        }
        self.imputer = {"mean": None, "most_frequent": None, "median": None}

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        # loop through the mean, mode, and median imputer
        for strategy, imputer in self.imputer.items():
            cols = self.impute_cols[strategy]
            if not cols:
                continue

            # use only the columns that exist in the data
            cols = list(set(cols).intersection(data.columns))

            if imputer is None:
                # create the imputer and impute the data
                imputer = SimpleImputer(strategy=strategy)
                data[cols] = imputer.fit_transform(data[cols])
                self.imputer[strategy] = imputer  # save the imputer
            else:
                # use existing imputer to impute the data
                data[cols] = imputer.transform(data[cols])
        return data


def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing data that can be filled heuristically"""
    # fill the following missing data with 0
    col = "num_prior_ED_visits_within_5_years"
    df[col] = df[col].fillna(0)

    # fill the following missing data with the maximum value
    for col in ["days_since_last_treatment", "days_since_prev_ED_visit"]:
        df[col] = df[col].fillna(df[col].max())

    return df


class OneHotEncoder:
    """One-hot encode (OHE) categorical data.

    Create separate indicator columns for each unique category and assign binary
    values of 1 or 0 to indicate the category's presence.
    """

    def __init__(self):
        self.encode_cols = ["regimen", "intent"]
        self.final_columns = None  # the final feature names after OHE

    def encode(self, data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        # one-hot encode categorical columns
        # use only the columns that exist in the data
        cols = [col for col in self.encode_cols if col in data.columns]
        data = pd.get_dummies(data, columns=cols)

        if self.final_columns is None:
            data = collapse_rare_categories(
                data, catcols=["regimen"]
            )  # collapse rare regimens into 'Other' category
            self.final_columns = data.columns
            return data

        # reassign any indicator columns that did not exist in final columns as other
        for feature in cols:
            indicator_cols = data.columns[data.columns.str.startswith(feature)]
            extra_cols = indicator_cols.difference(self.final_columns)
            if extra_cols.empty:
                continue

            if verbose:
                count = data[extra_cols].sum()
                msg = (
                    f"Reassigning the following {feature} indicator columns "
                    f"that did not exist in train set as other:\n{count}"
                )
                logger.info(msg)

            other_col = f"{feature}_other"
            if other_col not in data:
                data[other_col] = 0
            data[other_col] |= data[extra_cols].any(axis=1).astype(int)
            data = data.drop(columns=extra_cols)

        # fill in any missing columns
        missing_cols = self.final_columns.difference(data.columns)
        # use concat instead of data[missing_cols] = 0 to prevent perf warning
        data = pd.concat(
            [data, pd.DataFrame(0, index=data.index, columns=missing_cols)], axis=1
        )

        return data


class PrepData:
    """Prepare the data for model training"""

    def __init__(self):
        self.imp = Imputer()  # imputer
        self.ohe = OneHotEncoder()  # one-hot encoder
        self.scaler = None  # normalizer
        self.clip_thresh = None  # outlier clippers

        self.norm_cols = (
            [
                "height",
                "weight",
                "body_surface_area",
                "cycle_number",
                "age",
                "visit_month_sin",
                "visit_month_cos",
                "line_of_therapy",
                "days_since_starting_treatment",
                "days_since_last_treatment",
                "num_prior_EDs_within_5_years",
                "days_since_prev_ED",
                "patient_ecog",
                "patient_ecog_change",
            ]
            + symp_cols
            + lab_cols
            + drug_cols
            + lab_change_cols
            + symp_change_cols
        )
        self.clip_cols = (
            [
                "height",
                "weight",
                "body_surface_area",
            ]
            + lab_cols
            + lab_change_cols
        )

    def transform_data(
        self,
        data,
        one_hot_encode: bool = True,
        clip: bool = True,
        impute: bool = True,
        normalize: bool = True,
        ohe_kwargs: Optional[dict] = None,
        data_name: Optional[str] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Transform (one-hot encode, clip, impute, normalize) the data.

        Args:
            ohe_kwargs (dict): a mapping of keyword arguments fed into
                OneHotEncoder.encode

        IMPORTANT: always make sure train data is done first before valid
        or test data
        """
        if ohe_kwargs is None:
            ohe_kwargs = {}
        if data_name is None:
            data_name = "the"

        if one_hot_encode:
            # One-hot encode categorical data
            if verbose:
                logger.info(f"One-hot encoding {data_name} data")
            data = self.ohe.encode(data, **ohe_kwargs)

        if clip:
            # Clip the outliers based on the train data quantiles
            data = self.clip_outliers(data)

        if impute:
            # Impute missing data based on the train data mode/median/mean
            data = self.imp.impute(data)

        if normalize:
            # Scale the data based on the train data distribution
            data = self.normalize_data(data)

        return data

    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # use only the columns that exist in the data
        norm_cols = [col for col in self.norm_cols if col in data.columns]

        if self.scaler is None:
            self.scaler = StandardScaler()
            data[norm_cols] = self.scaler.fit_transform(data[norm_cols])
        else:
            data[norm_cols] = self.scaler.transform(data[norm_cols])
        return data

    def clip_outliers(
        self,
        data: pd.DataFrame,
        lower_percentile: float = 0.001,
        upper_percentile: float = 0.999,
    ) -> pd.DataFrame:
        """Clip the upper and lower percentiles for the columns indicated below"""
        # use only the columns that exist in the data
        cols = [col for col in self.clip_cols if col in data.columns]

        if self.clip_thresh is None:
            percentiles = [lower_percentile, upper_percentile]
            self.clip_thresh = data[cols].quantile(percentiles)

        data[cols] = data[cols].clip(
            lower=self.clip_thresh.loc[lower_percentile],
            upper=self.clip_thresh.loc[upper_percentile],
            axis=1,
        )
        return data
