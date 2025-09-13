"""
Module for preparation and training and evaluation pipelines
"""
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from make_clinical_dataset.epr.prep import Splitter
from ml_common.autogluon import evaluate, train_models
from sklearn.model_selection import StratifiedGroupKFold


def prepare(
    df: pd.DataFrame,
    n_folds: int = 5,
    split_date: str = "2022-01-01",
    target = "target_ED_30d",
) -> dict[str, pd.DataFrame]:
    """Split the data into input features, output targets, and meta info

    Assign cross-validation folds and data splits to each data sample, stored in meta info.
    """
    # split the data - create development (EPR) and test (EPIC) set
    splitter = Splitter()
    dev_data, test_data = splitter.temporal_split(df, split_date=split_date, visit_col="assessment_date")

    # split training data into folds for cross validation
    # NOTE: feel free to add more columns for different fold splits by looping through different random states
    kf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    kf_splits = kf.split(X=dev_data, y=dev_data[target], groups=dev_data["mrn"])
    cv_folds = np.zeros(len(dev_data))
    for fold, (_, valid_idxs) in enumerate(kf_splits):
        cv_folds[valid_idxs] = fold
    dev_data["cv_folds"] = cv_folds

    # create a split column and combine the data for convenience
    dev_data['split'], test_data['split'] = "Train", "Test"
    data = pd.concat([dev_data, test_data])

    # split into input features, output targets, and meta info
    meta_cols = [
        'mrn', 'assessment_date', 'split', 'cv_folds', 
        'primary_site_desc', 'drug_name', 'postal_code', 'target_ED_note', 
        'target_hemoglobin_min', 'target_platelet_min', 'target_neutrophil_min',
        'target_creatinine_max', 'target_alanine_aminotransferase_max',
        'target_aspartate_aminotransferase_max', 'target_total_bilirubin_max',
    ]
    targ_cols = [col for col in df.columns if col.startswith('target') and col not in meta_cols]
    feat_cols = data.columns.drop(meta_cols+targ_cols).tolist()
    return {
        'feats': data[feat_cols], 
        'targs': data[targ_cols], 
        'meta': data[meta_cols]
    }


def train_and_eval(
    out: dict[str, pd.DataFrame], 
    targets: list[str], 
    save_path: str, 
    load_model: bool = False,
    train_kwargs: dict | None = None,
    eval_kwargs: dict | None = None
) -> dict[str]:
    """
    Args:
        out: The output from preduce.acu.pipeline.prepare
        **kwargs: The keyword arguments for ml_common.autogluon.train_models
    """
    if train_kwargs is None:
        train_kwargs = {}
    if eval_kwargs is None:
        eval_kwargs = {}

    feats, targs, meta = out['feats'], out['targs'], out['meta']
    dev, test = meta["split"] == "Train", meta["split"] == "Test"
    
    if load_model:
        # Load the models
        models = {}
        for target in targets:
            models[target] = TabularPredictor.load(f'{save_path}/{target}', verbosity=0)
    else:
        # Train the models
        models = train_models(
            feats[dev], 
            targs[dev][targets],
            meta[dev], 
            save_path=save_path,
            **train_kwargs
        )

    # Get model performance in validation set
    val_score = {}
    for target in models:
        val_score[target] = models[target].leaderboard()[['model', 'score_val']]
    val_score = pd.concat(val_score, axis=1)

    # Get model performance in test set
    test_score = evaluate(models, feats[test], target[test], **eval_kwargs)

    return {"models": models, "val": val_score, "test": test_score}