"""
Module for final data preparation pipelines
"""

from warnings import simplefilter

import numpy as np
import pandas as pd
from make_clinical_dataset.epr.engineer import (
    collapse_rare_categories,
    get_change_since_prev_session,
    get_missingness_features,
)
from make_clinical_dataset.epr.filter import (
    drop_highly_missing_features,
    drop_samples_outside_study_date,
    drop_unused_drug_features,
    keep_only_one_per_week,
)
from make_clinical_dataset.epr.prep import (
    PrepData,
    Splitter,
    fill_missing_data_heuristically,
)
from make_clinical_dataset.epr.util import get_excluded_numbers
from sklearn.model_selection import StratifiedGroupKFold

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class PrepACUData(PrepData):
    def preprocess(
        self,
        df: pd.DataFrame,
        drop_cols_missing_thresh: int = 80,
        drop_rows_missing_thresh: int = 80,
        start_date: str = "2012-01-01",
        end_date: str = "2019-12-31",
    ) -> pd.DataFrame:
        """
        Args:
            drop_cols_missing_thresh: the percentage of missingness in which a column would be dropped.
                If set to -1, no columns will be dropped
            drop_rows_missing_thresh: the percentage of missingness in which a row would be dropped.
                If set to -1, no rows will be dropped
        """
        # convert cancer site and morphology features to binary variables
        # by taking the most recent diagnosis prior to assessment date, represented as 2
        cols = df.columns[df.columns.str.contains("cancer_site_|morphology_")]
        df[cols] = df[cols] == 2

        # keep only the first treatment session of a given week
        df = keep_only_one_per_week(df)

        # get the change in measurement since previous assessment
        df = get_change_since_prev_session(df)

        # filter out dates before 2012 and after 2020
        df = drop_samples_outside_study_date(
            df, start_date=start_date, end_date=end_date
        )

        # drop drug features that were never used
        df = drop_unused_drug_features(df)

        # fill missing data that can be filled heuristically (zeros, max values, etc)
        df = fill_missing_data_heuristically(df)

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
        self,
        df: pd.DataFrame,
        n_folds: int = 3,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # split the data - create training and testing set
        splitter = Splitter()
        train_data, test_data = splitter.temporal_split(
            df, split_date="2018-02-01", visit_col="assessment_date"
        )
        
        # IMPORTANT: always make sure train data is done first for one-hot encoding, clipping, imputing, scaling
        train_data = self.transform_data(train_data, data_name="training")
        test_data = self.transform_data(test_data, data_name="testing")

        # split training data into folds for cross validation
        # NOTE: feel free to add more columns for different fold splits by looping through different random states
        kf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        kf_splits = kf.split(
            X=train_data,
            y=train_data["target_ED_30d"],  # placeholder
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
        Y.columns = Y.columns.str.replace("target_", "")

        return X, Y, metainfo
