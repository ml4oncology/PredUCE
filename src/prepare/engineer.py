"""
Module to engineer features
"""
from collections.abc import Sequence

from tqdm import tqdm
import numpy as np
import pandas as pd

from .. import logger
from ..constants import lab_cols, lab_change_cols, symp_cols, symp_change_cols

###############################################################################
# Engineering Features
###############################################################################
def get_change_since_prev_session(df: pd.DataFrame) -> pd.DataFrame:
    """Get change since last session"""
    cols = symp_cols + lab_cols + ['patient_ecog']
    change_cols = symp_change_cols + lab_change_cols + ['patient_ecog_change']
    result = []
    for mrn, group in tqdm(df.groupby('mrn')):
        change = group[cols] - group[cols].shift()
        result.append(change.reset_index().to_numpy())
    result = np.concatenate(result)

    result = pd.DataFrame(result, columns=['index']+change_cols).set_index('index')
    result.index = result.index.astype(int)
    df = pd.concat([df, result], axis=1)

    return df


def get_missingness_features(df: pd.DataFrame, target_keyword: str = 'target') -> pd.DataFrame:
    cols_with_nan = df.columns[df.isnull().any()]
    cols_with_nan = cols_with_nan[~cols_with_nan.str.startswith(target_keyword)]
    df[cols_with_nan + '_is_missing'] = df[cols_with_nan].isnull()
    return df


def collapse_rare_categories(df: pd.DataFrame, catcols: Sequence[str]) -> pd.DataFrame:
    """Collapse rare categories to 'Other'"""
    for feature in catcols:
        other_mask = False
        drop_cols = []
        for col in df.columns[df.columns.str.startswith(feature)]:
            mask = df[col]
            if df.loc[mask, 'mrn'].nunique() < 6:
                drop_cols.append(col)
                other_mask |= mask
        df = df.drop(columns=drop_cols)
        df[f'{feature}_other'] = other_mask
        msg = f'Reassigning the following {len(drop_cols)} indicators with less than 6 patients as other: {drop_cols}'
        logger.info(msg)
    return df