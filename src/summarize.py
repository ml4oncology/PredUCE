"""
Module to create summary tables
"""
import pandas as pd

def feature_summary(X_train: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    """
    Args:
        X_train (pd.DataFrame): table of the original data (not one-hot 
            encoded, normalized, clipped, etc) for the training set
    """
    N = len(X_train)

    # remove missingness features
    cols = X_train.columns
    drop_cols = cols[cols.str.contains('is_missing')]
    X_train = X_train.drop(columns=drop_cols)

    # get number of missing values, mean, and standard deviation for each 
    # feature for the training set
    summary = X_train.astype(float).describe()
    summary = summary.loc[['count', 'mean', 'std']].T
    summary = summary.round(6)
    summary['count'] = N - summary['count']
    # mask small cells less than 6
    summary['count'] = summary['count'].replace({i:6 for i in range(1,6)})
    summary['Missingness (%)'] = (summary['count'] / N * 100).round(1)
    
    format_arr = lambda arr: arr.round(3).astype(str)
    mean, std = format_arr(summary['mean']), format_arr(summary['std'])
    summary['Mean (SD)'] = mean + ' (' + std + ')'
    
    name_map = {
        'count': f'Train (N={N}) - Missingness Count', 
        'mean': 'Train - Mean', 
        'std': 'Train - SD'
    }
    summary = summary.rename(columns=name_map)

    # assign the groupings for each feature
    # features = summary.index
    # for group, keyword in variable_groupings_by_keyword.items():
    #     summary.loc[features.str.contains(keyword), 'Group'] = group
    
    # insert units
    # rename_map = {name: f'{name} ({unit})' for name, unit in get_units().items()}
    # summary = summary.rename(index=rename_map)
    
    # summary.index = get_clean_variable_names(summary.index)
    summary.to_csv(f'{save_dir}/feature_summary.csv')
    return summary