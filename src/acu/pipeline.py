"""
Module for final data preparation pipelines
"""

import pandas as pd

from ml_common.engineer import (
    collapse_rare_categories,
    get_change_since_prev_session,
    get_missingness_features,
)
from ml_common.filter import drop_highly_missing_features, drop_unused_drug_features
from ml_common.prep import PrepData, Splitter
from ..prepare.filter import drop_samples_outside_study_date, exclude_immediate_events
from ..prepare.prep import fill_missing_data
from .label import get_event_labels

from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class PrepACUData(PrepData):
    def preprocess(self, df: pd.DataFrame, emerg: pd.DataFrame) -> pd.DataFrame:
        # keep only the first treatment session of a given week
        # df = keep_only_one_per_week(df)
        # get the change in measurement since previous assessment
        df = get_change_since_prev_session(df)
        # extract labels
        df = get_event_labels(
            df,
            emerg,
            event_name="ED_visit",
            extra_cols=["CTAS_score", "CEDIS_complaint"],
        )
        # filter out dates before 2014 and after 2020
        df = drop_samples_outside_study_date(df)
        # drop drug features that were never used
        df = drop_unused_drug_features(df)
        # fill missing data that can be filled heuristically
        df = fill_missing_data(df)
        # drop features with high missingness
        keep_cols = df.columns[df.columns.str.contains("target_")]
        df = drop_highly_missing_features(df, missing_thresh=80, keep_cols=keep_cols)
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
        train_data, valid_data, test_data = splitter.split_data(
            df, split_date="2018-02-01"
        )

        # Remove sessions where event occured immediately afterwards on the train and valid set ONLY
        train_data = exclude_immediate_events(
            train_data, date_cols=["target_ED_visit_date"]
        )
        valid_data = exclude_immediate_events(
            valid_data, date_cols=["target_ED_visit_date"]
        )

        # IMPORTANT: always make sure train data is done first for one-hot encoding, clipping, imputing, scaling
        train_data = self.transform_data(train_data, data_name="training")
        valid_data = self.transform_data(valid_data, data_name="validation")
        test_data = self.transform_data(test_data, data_name="testing")

        # create a split column and combine the data for convenience
        train_data[["cohort", "split"]] = ["Development", "Train"]
        valid_data[["cohort", "split"]] = ["Development", "Valid"]
        test_data[["cohort", "split"]] = "Test"
        data = pd.concat([train_data, valid_data, test_data])

        # split into input features, output labels, and metainfo
        cols = data.columns
        meta_cols = ["mrn", "cohort", "split"] + cols[
            cols.str.contains("date")
        ].tolist()
        targ_cols = cols[
            cols.str.contains("target") & ~cols.str.contains("date")
        ].tolist()
        feat_cols = cols.drop(meta_cols + targ_cols).tolist()
        X, Y, metainfo = data[feat_cols], data[targ_cols], data[meta_cols]
        return X, Y, metainfo
