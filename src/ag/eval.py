"""
Module to evaluate models with Autogluon
"""

from typing import Optional

import pandas as pd
from autogluon.tabular import TabularPredictor


###############################################################################
# Evaluation
###############################################################################
def evaluate(
    models: dict[str, TabularPredictor],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    return_full: bool = False,
) -> pd.DataFrame:
    """Evaluate performance for all targets and all model types

    Args:
        return_full: If True, return the full information about models (training times, inference times, stack levels, etc)
    """
    results = {}
    for target, model in models.items():
        data = pd.concat([X, Y[target]], axis=1)
        res = model.leaderboard(data, extra_metrics=["roc_auc", "average_precision"])
        results[target] = (
            res if return_full else res[["model", "roc_auc", "average_precision"]]
        )
    results = pd.concat(results, axis=1)
    return results


###############################################################################
# Predictions
###############################################################################
def get_val_pred(model: TabularPredictor, model_name: Optional[str] = None):
    """Get the predictions in the validation set"""
    if model._trainer.bagged_mode:  # used cross-validation
        return model.predict_proba_oof(model=model_name)[True]  # oof = out of fold
    else:
        x, y = model.load_data_internal(data="val")
        return model.predict_proba(x, model=model_name)[True]


def get_all_val_pred(models: dict[str, TabularPredictor]):
    """Get the predictions in the validation set for all targets and all model types"""
    res = {}
    for target, model in models.items():
        preds = {}
        for model_name in model.model_names():
            preds[model_name] = get_val_pred(model, model_name=model_name)
        res[target] = pd.DataFrame(preds).dropna().reset_index()
    res = pd.concat(res, axis=1)
    return res
