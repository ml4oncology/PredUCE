"""
Module to train models with Autogluon
"""

from typing import Optional

from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularPredictor
from datetime import datetime
import numpy as np
import pandas as pd

from ml_common.eval import auc_scores


###############################################################################
# Training
###############################################################################
def train_models(
    X: pd.DataFrame, Y: pd.DataFrame, metainfo: pd.DataFrame, **kwargs
) -> dict[str, TabularPredictor]:
    models = {}
    for target, label in Y.items():
        data = pd.concat([X, label, metainfo["cv_folds"]], axis=1)
        models[target] = train_model(data, target, **kwargs)
    return models


def train_model(
    data: pd.DataFrame,
    target: str,
    eval_metric: str = "average_precision",
    presets: str = "medium_quality",
    calibrate: bool = False,
    refit_on_full_data: bool = False,
    time_limit: int = 10000,  # seconds
    save_path: Optional[str] = None,
    extra_init_kwargs: Optional[dict] = None,
    extra_fit_kwargs: Optional[dict] = None,
) -> TabularPredictor:
    """
    Args:
        refit_on_full_data: If True, refit the model with the full training dataset at the end.
            Note the only difference between 'high' and 'best' preset is that 'high' refits on the full data, 'best'
            does not (as of 2024-12-17)
    """
    if eval_metric == "auc_combo":
        eval_metric = auc_combo_score
    if save_path is None:
        time = datetime.now().strftime(format="%Y%m%d_%H%M%S")
        quality = presets.replace("_quality", "")
        save_path = f"AutogluonModels/{time}-{target}-{quality}-{eval_metric}"
    if extra_init_kwargs is None:
        extra_init_kwargs = {}
    if extra_fit_kwargs is None:
        extra_fit_kwargs = {}

    # set up the training parameters
    init_kwargs = dict(
        log_to_file=True, path=save_path, eval_metric=eval_metric, **extra_init_kwargs
    )
    fit_kwargs = dict(
        presets=presets,
        # feature_prune_kwargs={}, # mixed results with feature pruning
        excluded_model_types=[
            "FASTAI",
            "NN_TORCH",
        ],  # they perform badly anyways, on top them being slow
        # included_model_types=['XGB'],
        # fit_weighted_ensemble=False,
        calibrate=calibrate,
        save_bag_folds=True,  # save the individual cross validation fold models
        time_limit=time_limit,
        refit_full=refit_on_full_data,  # refit the model on all of the data in the end
        set_best_to_refit_full=refit_on_full_data,
        **extra_fit_kwargs,
    )

    if quality == "medium":
        # not using cross-validation, use the following as the tuning set
        mask = data.pop("cv_folds") == 0
        fit_kwargs["tuning_data"] = data[mask]
        data = data[~mask]
    else:
        # use our own cross validation folds
        init_kwargs["groups"] = "cv_folds"

    print(init_kwargs, fit_kwargs)

    predictor = TabularPredictor(label=target, **init_kwargs).fit(data, **fit_kwargs)
    return predictor


###############################################################################
# Scoring
###############################################################################
def auc_combo(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Combination of AUROC and AUPRC"""
    aucs = auc_scores(y_pred, y_true)
    return aucs["AUROC"] + aucs["AUPRC"]


auc_combo_score = make_scorer(
    name="auc_combo", score_func=auc_combo, greater_is_better=True
)
