"""
Module to create summary tables
"""
from typing import Optional

import pandas as pd

from .constants import cancer_code_map, drug_cols, lab_cols, unit_map


def pre_and_post_treatment_missingness_summary(
    df: pd.DataFrame, 
    pretreatment_cols: list[str], 
    posttreatment_cols: list[str], 
    event_cols: list[str], 
    print_desc: bool = True
) -> pd.DataFrame:
    """Creates a summary table of the following:

        Missingness_b: proportion of treatments where the post-treatment target is measured but pre-treatment feature is not measured
        Missingness_c: proportion of treatments where the pre-treatment feature is measured but the post-treatment target is not measured
        Rate_d: proportion of treatments followed by the event, when the pre-treatment feature and post-treatment target is measured
            - well, by default both must be measured if the event occured
        Missingness_e: the proportion of patients with at least one no target measurement
        Rate_f: proportion of patients with at least one treatment followed by an event, among patients who have at least one target measurement.
    """
    n_total_patients = df['mrn'].nunique()
    result = {}
    for pre_col, post_col, event_col in zip(pretreatment_cols, posttreatment_cols, event_cols):
        posttreatment_exists = df[pre_col].notnull()
        pretreatment_exists = df[post_col].notnull()
        event_occured = df[event_col] == 1
        result[pre_col] = {
            'Missingness_b': (posttreatment_exists & ~pretreatment_exists).sum(), 
            'Missingness_c': ( ~posttreatment_exists & pretreatment_exists).sum(), 
            'Rate_d': event_occured.sum(), 
            'Missingness_e': (~posttreatment_exists).groupby(df['mrn']).any().sum(), 
            'num_patients_with_event': df.loc[event_occured, 'mrn'].nunique(), 
            'num_patients_with_post': df.loc[posttreatment_exists, 'mrn'].nunique()
        }
    result = pd.DataFrame(result).T
    result.loc['Mean'] = result.mean().astype(int)

    # add the percentages
    add_perc = lambda df, N: df.astype(str) + ' (' + (df / N * 100).round(1).astype(str) + ')'
    result[['Missingness_b', 'Missingness_c', 'Rate_d']] = add_perc(result[['Missingness_b', 'Missingness_c', 'Rate_d']], N=len(df))
    result['Missingness_e'] = add_perc(result['Missingness_e'], N=n_total_patients)
    result['Rate_f'] = add_perc(result['num_patients_with_event'], N=result['num_patients_with_post'])
    result = result.drop(columns=['num_patients_with_event', 'num_patients_with_post'])

    # add a second level
    result.columns = [[f'Treatment (N={len(df)})',]*3 + [f'Patient level (N={n_total_patients})',]*2, result.columns]

    if print_desc:
        print("Missingness_b: proportion of treatments where the post-treatment target is measured but pre-treatment feature is not measured")
        print("Missingness_c: proportion of treatments where the pre-treatment feature is measured but the post-treatment target is not measured")
        print("Rate_d: proportion of treatments followed by the event, when the pre-treatment feature and post-treatment target is measured")
        print("Missingness_e: the proportion of patients with at least one no target measurement")
        print("Rate_f: proportion of patients with at least one treatment followed by an event, among patients who have at least one target measurement.")
    return result


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