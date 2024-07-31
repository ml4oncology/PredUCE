"""
Module to evaluate models
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def predict(models, data):
    # average across the folds
    return np.mean([m.predict_proba(data)[:, 1] for m in models], axis=0)


def evaluate_test(model, X, Y):
    result = {}
    for target, label in Y.items():
        pred = predict(model[target], X)
        result[target] = {
            "AUPRC": average_precision_score(label, pred),
            "AUROC": roc_auc_score(label, pred),
        }
    return pd.DataFrame(result)


def evaluate_valid(
    models: dict, X: pd.DataFrame, Y: pd.DataFrame, metainfo: pd.DataFrame
):
    result = {}
    for alg, estimators in models.items():
        output = _evaluate_all_targets(estimators, X, Y, metainfo)
        result[alg] = pd.DataFrame(output)
    return pd.concat(result).T


def _evaluate_all_targets(
    models: dict, X: pd.DataFrame, Y: pd.DataFrame, metainfo: pd.DataFrame
):
    """Evaluate models for each target by averaging the performance across cross-validation folds"""
    result = {}
    for target, label in Y.items():
        output = _evaluate_across_folds(models[target], X, label, metainfo)
        result[target] = pd.DataFrame(output).mean(axis=1)
    return result


def _evaluate_across_folds(
    models: list, X: pd.DataFrame, Y: pd.Series, metainfo: pd.DataFrame
):
    """Evaluate models for each fold using it's associated validation set"""
    result = {}
    for fold, model in enumerate(models):
        mask = metainfo["cv_folds"] == fold
        X_valid, Y_valid = X[mask], Y[mask]
        result[fold] = _evaluate(model, X_valid, Y_valid)
    return result


def _evaluate(model, data: pd.DataFrame, label: pd.Series):
    pred = model.predict_proba(data)[:, 1]
    return {
        "AUPRC": average_precision_score(label, pred),
        "AUROC": roc_auc_score(label, pred),
    }
