from collections.abc import Sequence

import pandas as pd

from ..constants import symp_cols
from .filter import exclude_immediate_events, indicate_immediate_events
from .prep import PrepData
from .split import create_train_val_test_splits

from warnings import simplefilter 
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

###############################################################################
# Symptom
###############################################################################
def symptom_prep_pipeline(
    df: pd.DataFrame, 
    split_date: str, 
    target_pt_increases: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # split the data - create training, validation, testing set
    train_data, valid_data, test_data = create_train_val_test_splits(df, split_date=split_date)

    # Remove sessions where event occured immediately afterwards on the train and valid set ONLY
    date_cols = [f'target_{symp}_survey_date' for symp in symp_cols]
    for pt in target_pt_increases:
        targ_cols = [f'target_{symp}_{pt}pt_change' for symp in symp_cols]
        train_data = indicate_immediate_events(train_data, targ_cols, date_cols)
        valid_data = indicate_immediate_events(valid_data, targ_cols, date_cols)

    # IMPORTANT: always make sure train data is done first for one-hot encoding, clipping, imputing, scaling
    prep = PrepData()
    train_data = prep.transform_data(train_data, data_name='training')
    valid_data = prep.transform_data(valid_data, data_name='validation')
    test_data = prep.transform_data(test_data, data_name='testing')

    # create a split column and combine the data for convenience
    train_data[['cohort', 'split']] = ['Development', 'Train']
    valid_data[['cohort', 'split']] = ['Development', 'Valid']
    test_data[['cohort', 'split']] = 'Test'
    data = pd.concat([train_data, valid_data, test_data])

    # split into input features, output labels, and metainfo
    cols = data.columns
    meta_cols = ['mrn', 'cohort', 'split'] + cols[cols.str.contains('date')].tolist()
    targ_cols = cols[cols.str.contains('target') & ~cols.str.contains('date')].tolist()
    feat_cols = cols.drop(meta_cols+targ_cols).tolist()
    X, Y, metainfo = data[feat_cols], data[targ_cols], data[meta_cols]
    return X, Y, metainfo


###############################################################################
# Acute Care Use
###############################################################################
def acu_prep_pipeline(df: pd.DataFrame, event_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # split the data - create training, validation, testing set
    train_data, valid_data, test_data = create_train_val_test_splits(df, split_date='2018-02-01')

    # Remove sessions where event occured immediately afterwards on the train and valid set ONLY
    train_data = exclude_immediate_events(train_data, date_cols=['target_ED_visit_date'])
    valid_data = exclude_immediate_events(valid_data, date_cols=['target_ED_visit_date'])

    # IMPORTANT: always make sure train data is done first for one-hot encoding, clipping, imputing, scaling
    prep = PrepData()
    train_data = prep.transform_data(train_data, data_name='training')
    valid_data = prep.transform_data(valid_data, data_name='validation')
    test_data = prep.transform_data(test_data, data_name='testing')

    # create a split column and combine the data for convenience
    train_data[['cohort', 'split']] = ['Development', 'Train']
    valid_data[['cohort', 'split']] = ['Development', 'Valid']
    test_data[['cohort', 'split']] = 'Test'
    data = pd.concat([train_data, valid_data, test_data])

    # split into input features, output labels, and metainfo
    cols = data.columns
    meta_cols = ['mrn', 'cohort', 'split'] + cols[cols.str.contains('date')].tolist()
    targ_cols = cols[cols.str.contains('target') & ~cols.str.contains('date')].tolist()
    feat_cols = cols.drop(meta_cols+targ_cols).tolist()
    X, Y, metainfo = data[feat_cols], data[targ_cols], data[meta_cols]
    return X, Y, metainfo