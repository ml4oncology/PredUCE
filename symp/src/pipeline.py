"""
Module for final data preparation pipelines
"""

from collections.abc import Sequence
from warnings import simplefilter

import pandas as pd
from make_clinical_dataset.epr.prep import PrepData, Splitter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class PrepSympData(PrepData):
    def run_pipeline(
        self, df: pd.DataFrame, split_date: str, target_pt_increases: Sequence[int]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # split the data - create training, validation, testing set
        splitter = Splitter()
        train_data, valid_data, test_data = splitter.split_data(
            df, split_date=split_date
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
