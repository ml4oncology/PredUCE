"""
Module to create summary tables
"""
from typing import Optional

import pandas as pd

from .constants import cancer_code_map, drug_cols, lab_cols, unit_map

def feature_summary(
    X_train: pd.DataFrame, 
    save_path: Optional[str] = None, 
    keep_orig_names: bool = False,
    remove_missingness_feats: bool = True
) -> pd.DataFrame:
    """
    Args:
        X_train (pd.DataFrame): table of the original data (not one-hot encoded, normalized, clipped, etc) for the 
            training set
    """
    N = len(X_train)

    if remove_missingness_feats:
        # remove missingness features
        cols = X_train.columns
        drop_cols = cols[cols.str.contains('is_missing')]
        X_train = X_train.drop(columns=drop_cols)

    # get number of missing values, mean, and standard deviation for each feature in the training set
    summary = X_train.astype(float).describe()
    summary = summary.loc[['count', 'mean', 'std']].T
    count = N - summary['count']
    mean = summary['mean'].round(3).apply(lambda x: f'{x:.3f}')
    std = summary['std'].round(3).apply(lambda x: f'{x:.3f}')
    count[count.between(1, 5)] = 6 # mask small cells less than 6
    summary['Mean (SD)'] = mean + ' (' + std + ')'
    summary['Missingness (%)'] = (count / N * 100).round(1)
    summary = summary.drop(columns=['count', 'mean', 'std'])
    # special case for drug features (percentage of dose given)
    for col in drug_cols:
        if col not in X_train.columns: continue
        mask = X_train[col] != 0 # 0 indicates no drugs were given (not the percentage of the given dose)
        vals = X_train.loc[mask, col]
        summary.loc[col, 'Mean (SD)'] = f'{vals.mean():.3f} ({vals.std():.3f})'

    # assign the groupings for each feature
    feature_groupings_by_keyword = {
        'Acute care use': 'ED_visit',
        'Cancer': 'cancer_site|morphology', 
        'Demographic': 'height|weight|body_surface_area|female|age',
        'Laboratory': '|'.join(lab_cols),
        'Treatment': 'visit_month|regimen|intent|treatment|dose|therapy|cycle',
        'Symptoms': 'esas|ecog'
    }
    features = summary.index
    for group, keyword in feature_groupings_by_keyword.items():
        summary.loc[features.str.contains(keyword), 'Group'] = group
    summary = summary[['Group', 'Mean (SD)', 'Missingness (%)']]
    
    if keep_orig_names: 
        summary['Features (original)'] = summary.index

    # insert units
    rename_map = {feat: f'{feat} ({unit})' for unit, feats in unit_map.items() for feat in feats}
    rename_map['female'] = 'female (yes/no)'
    summary = summary.rename(index=rename_map)
    
    summary.index = [clean_feature_name(feat) for feat in summary.index]
    summary = summary.reset_index(names='Features')
    summary = summary.sort_values(by=['Group', 'Features'])
    if save_path is not None: summary.to_csv(f'{save_path}', index=False)
    return summary


def clean_feature_name(name: str) -> str:
    if name == 'patient_ecog':
        return 'Eastern Cooperative Oncology Group (ECOG) Performance Status'
    
    mapping = {
        'prev': 'previous', 
        'num_': 'number_of_', 
        '%_ideal_dose': 'percentage_of_ideal_dose',
        'intent': 'intent_of_systemic_treatment', 
        'cancer_site': 'topography_ICD-0-3', 
        'morphology': 'morphology_ICD-0-3', 
        'shortness_of_breath': 'dyspnea',
        'tiredness': 'fatigue',
        'patient_ecog': 'eastern_cooperative_oncology_group_(ECOG)_performance_status',
        'cycle_number': 'chemotherapy_cycle',
    }
    for orig, new in mapping.items():
        name = name.replace(orig, new)

    # title the name and replace underscores with space, but don't modify anything inside brackets at the end
    if name.endswith(')') and not name.startswith('regimen'):
        name, extra_info = name.split('(')
        name = '('.join([name.replace('_', ' ').title(), extra_info])
    else:
        name = name.replace('_', ' ').title()

    # capitalize certain substrings
    for substr in ['Ed V', 'Icd', 'Other', 'Esas', 'Ecog']:
        name = name.replace(substr, substr.upper())
    # lowercase certain substrings
    for substr in [' Of ']:
        name = name.replace(substr, substr.lower())

    if name.startswith('Topography ') or name.startswith('Morphology '):
        # get full cancer description 
        code = name.split(' ')[-1]
        if code in cancer_code_map:
            name = f"{name}, {cancer_code_map[code]}"
    elif name.startswith('ESAS '):
        # add 'score'
        if 'Change' in name: name = name.replace('Change', 'Score Change')
        else: name += ' Score'

    for prefix in ['Regimen ', 'Percentage of Ideal Dose Given ']:
        if name.startswith(prefix):
            # capitalize all regimen / drug names
            name = f"{prefix}{name.split(prefix)[-1].upper()}"

    return name