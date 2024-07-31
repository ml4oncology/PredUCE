"""
Module for final data preparation pipelines
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ml_common.engineer import (
    collapse_rare_categories,
    get_change_since_prev_session,
    get_missingness_features,
)
from ml_common.filter import drop_highly_missing_features, drop_unused_drug_features
from ml_common.prep import PrepData, Splitter
from ml_common.util import get_excluded_numbers
from ..prepare.filter import (
    drop_samples_outside_study_date,
    exclude_immediate_events,
    keep_only_one_per_week,
)
from ..prepare.prep import fill_missing_data
from .label import get_event_labels

from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class PrepACUData(PrepData):
    def preprocess(
        self,
        df: pd.DataFrame,
        emerg: pd.DataFrame,
        drop_cols_missing_thresh: int = 80,
        drop_rows_missing_thresh: int = 70,
    ) -> pd.DataFrame:
        """
        Args:
            drop_cols_missing_thresh: the percentage of missingness in which a column would be dropped.
                If set to -1, no columns will be dropped
            drop_rows_missing_thresh: the percentage of missingness in which a row would be dropped.
                If set to -1, no rows will be dropped
        """
        # keep only the first treatment session of a given week
        df = keep_only_one_per_week(df)

        # get the change in measurement since previous assessment
        df = get_change_since_prev_session(df)

        # extract labels
        df = get_event_labels(
            df,
            emerg,
            event_name="ED_visit",
            extra_cols=["CTAS_score", "CEDIS_complaint"],
        )

        # filter out dates before 2012 and after 2020
        df = drop_samples_outside_study_date(
            df, start_date="2012-01-01", end_date="2019-12-31"
        )

        # drop drug features that were never used
        df = drop_unused_drug_features(df)

        # fill missing data that can be filled heuristically
        df = fill_missing_data(df)

        # To align with EPIC system for silent deployment
        #   1. remove drug and morphology features
        #   2. restrict to GI patients
        # This will be temporary
        df = df.loc[:, ~df.columns.str.contains("morphology|%_ideal_dose")]
        mask = df["regimen"].str.startswith("GI-")
        get_excluded_numbers(df, mask, context=" not from GI department")
        df = df[mask]

        if drop_cols_missing_thresh != -1:
            # drop features with high missingness
            keep_cols = df.columns[df.columns.str.contains("target_")]
            df = drop_highly_missing_features(
                df, missing_thresh=drop_cols_missing_thresh, keep_cols=keep_cols
            )

        if drop_rows_missing_thresh != -1:
            # drop samples with high missingness
            keep_cols = df.columns[~df.columns.str.contains("target_|date|mrn")]
            tmp = df[keep_cols].copy()
            # temporarily reverse the encoding for cancer-site
            cancer_site_cols = tmp.columns[tmp.columns.str.contains("cancer_site")]
            tmp["cancer_site"] = tmp[cancer_site_cols].apply(
                lambda mask: ", ".join(
                    cancer_site_cols[mask].str.removeprefix("cancer_site_")
                ),
                axis=1,
            )
            tmp["cancer_site"] = tmp["cancer_site"].replace("", None)
            tmp = tmp.drop(columns=cancer_site_cols)
            mask = tmp.isnull().mean(axis=1) * 100 < drop_rows_missing_thresh
            get_excluded_numbers(
                df,
                mask,
                context=f" with at least {drop_rows_missing_thresh} percent of features missing",
            )
            df = df[mask]

        # create missingness features
        df = get_missingness_features(df)

        # collapse rare morphology and cancer sites into 'Other' category
        df = collapse_rare_categories(df, catcols=["cancer_site", "morphology"])

        return df

    def prepare(
        self, df: pd.DataFrame, event_name: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # split the data - create training, validation, testing set
        splitter = Splitter()
        train_data, test_data = splitter.temporal_split(
            df, split_date="2018-02-01", visit_col="assessment_date"
        )

        # Remove sessions where event occured immediately afterwards on the train and valid set ONLY
        train_data = exclude_immediate_events(
            train_data, date_cols=["target_ED_visit_date"]
        )

        # If there are no prior values for height, weight, body surface area, take the median based on sex
        # TODO: maybe support this in ml_common.prep.imputer?
        mes_median = (
            train_data.groupby("female")[["height", "weight", "body_surface_area"]]
            .median()
            .T
        )
        mes_median.columns = ["F", "M"]

        def impute_mes(data):
            female = data["female"]
            data[female] = data[female].fillna(mes_median["F"])
            data[~female] = data[~female].fillna(mes_median["M"])
            return data

        train_data = impute_mes(train_data)
        test_data = impute_mes(test_data)

        # IMPORTANT: always make sure train data is done first for one-hot encoding, clipping, imputing, scaling
        train_data = self.transform_data(train_data, data_name="training")
        test_data = self.transform_data(test_data, data_name="testing")

        # split training data into folds for cross validation
        # NOTE: feel free to add more columns for different fold splits by looping through different random states
        kf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        kf_splits = kf.split(
            X=train_data,
            y=train_data["target_ED_visit"],  # placeholder
            groups=train_data["mrn"],
        )
        cv_folds = np.zeros(len(train_data))
        for fold, (_, valid_idxs) in enumerate(kf_splits):
            cv_folds[valid_idxs] = fold
        train_data["cv_folds"] = cv_folds

        # create a split column and combine the data for convenience
        train_data["split"], test_data["split"] = "Train", "Test"
        data = pd.concat([train_data, test_data])

        # split into input features, output labels, and metainfo
        cols = data.columns
        meta_cols = ["mrn", "split", "cv_folds"] + cols[
            cols.str.contains("date")
        ].tolist()
        targ_cols = cols[
            cols.str.contains("target_") & ~cols.str.contains("date")
        ].tolist()
        feat_cols = cols.drop(meta_cols + targ_cols).tolist()
        X, Y, metainfo = (
            data[feat_cols].copy(),
            data[targ_cols].copy(),
            data[meta_cols].copy(),
        )

        # clean up Y
        for col in ["target_CEDIS_complaint", "target_CTAS_score"]:
            metainfo[col] = Y.pop(col)
        Y.columns = Y.columns.str.replace("target_", "")

        return X, Y, metainfo
