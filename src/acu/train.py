"""
Module to train and tune the models using K-fold cross validation
"""

import warnings
import multiprocessing as mp
from functools import partial

import lightgbm as lgb
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .constants import bayesopt_param, model_static_param, model_tuning_param

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

algs = {
    "Ridge": LogisticRegression,
    "LASSO": LogisticRegression,
    "XGB": XGBClassifier,
    "LGBM": LGBMClassifier,
    "RF": RandomForestClassifier,
    # 'SVC': SVC
}


###############################################################################
# Training
###############################################################################
def train_model(
    X: pd.DataFrame, Y: pd.DataFrame, mrns: pd.Series, alg: str, best_params: dict
):
    models = {}
    for target, label in Y.items():
        kwargs = {**model_static_param[alg], **best_params[alg]}
        if alg in ["XGB", "LGBM"]:
            kwargs["scale_pos_weight"] = sum(label == 0) / sum(label == 1)

        models[target] = cross_validate(X, label, mrns, alg, **kwargs)

    return models


def train_models(X: pd.DataFrame, Y: pd.DataFrame, mrns: pd.Series, best_params: dict):
    return {alg: train_model(X, Y, mrns, alg, best_params) for alg in algs}


def cross_validate(
    X: pd.DataFrame,
    Y: pd.Series,
    mrns: pd.Series,
    alg: str,
    return_valid_data: bool = False,
    **kwargs,
):
    models, valid_data = [], []
    kf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train_idxs, valid_idxs) in enumerate(kf.split(X, Y, mrns)):
        # get the data splits
        X_train, X_valid = X.iloc[train_idxs].copy(), X.iloc[valid_idxs].copy()
        Y_train, Y_valid = Y.iloc[train_idxs].copy(), Y.iloc[valid_idxs].copy()

        # train the model
        if alg == "XGB":
            kwargs["early_stopping_rounds"] = 10
            fit_kwargs = {
                "eval_set": [(X_valid, Y_valid)],
                "verbose": 0,
            }
        elif alg == "LGBM":
            fit_kwargs = {
                "eval_set": [(X_valid, Y_valid)],
                "callbacks": [lgb.early_stopping(stopping_rounds=10, verbose=False)],
            }
        else:
            fit_kwargs = {}
        model = algs[alg](**kwargs)
        model.fit(X_train, Y_train, **fit_kwargs)
        models.append(model)
        valid_data.append((X_valid, Y_valid))

    if return_valid_data:
        return models, valid_data

    return models


###############################################################################
# Hyperparameter tuning
###############################################################################
def tune_params(
    alg: str, X: pd.DataFrame, Y: pd.Series, mrns: pd.Series, verbose: int = 2
):
    """Tunes hyperparameters for a given algorithm using Bayesian Optimization."""
    hyperparam_config = model_tuning_param[alg]
    data = (X, Y, mrns)
    bo = BayesianOptimization(
        f=partial(eval_func, alg=alg, data=data),
        pbounds=hyperparam_config,
        verbose=verbose,
        random_state=42,
        allow_duplicate_points=True,
    )

    # log the progress
    logger1 = JSONLogger(path=f"./logs/bayes_opt/{alg}-{Y.name}-bayesopt.log")
    logger2 = ScreenLogger(verbose=verbose, is_constrained=False)
    for bo_logger in [logger1, logger2]:
        bo.subscribe(Events.OPTIMIZATION_START, bo_logger)
        bo.subscribe(Events.OPTIMIZATION_STEP, bo_logger)
        bo.subscribe(Events.OPTIMIZATION_END, bo_logger)

    bo.maximize(**bayesopt_param[alg])
    best_param = bo.max["params"]

    return convert_params(best_param)


def convert_params(params):
    int_params = [
        "n_estimators",
        "max_depth",
        "num_leaves",
        "min_child_weight",
        "min_data_in_leaf",
        "min_samples_leaf",
        "bagging_freq",
    ]
    svc_kernels = ["linear", "poly", "rbf", "sigmoid"]
    for param, value in params.items():
        if param in int_params:
            params[param] = int(value)
        if param == "kernel":
            params[param] = svc_kernels[int(value)]
    return params


def eval_func(alg: str, data: tuple[pd.DataFrame, pd.Series, pd.Series], **kwargs):
    X, Y, mrns = data

    kwargs = {**model_static_param[alg], **convert_params(kwargs)}
    models, valid_data = cross_validate(
        X, Y, mrns, alg, return_valid_data=True, **kwargs
    )

    result = []
    for model, (X_valid, Y_valid) in zip(models, valid_data):
        assert model.classes_[1] == 1  # positive class is at index 1
        pred_prob = model.predict_proba(X_valid)[:, 1]
        result.append(roc_auc_score(Y_valid, pred_prob))

    return np.mean(result)


###############################################################################
# Feature Selection
###############################################################################
def feature_selection_by_lasso(X: pd.DataFrame, Y: pd.Series, mrns: pd.Series):
    kwargs = {"C": 0.1, **model_static_param["LASSO"]}
    models, valid_data = cross_validate(
        X, Y, mrns, alg="LASSO", return_valid_data=True, **kwargs
    )

    results, weights = {}, {}
    for fold, (model, (X_valid, Y_valid)) in enumerate(zip(models, valid_data)):
        pred_prob = model.predict_proba(X_valid)[:, 1]
        results[f"Fold {fold+1}"] = {"AUROC": roc_auc_score(Y_valid, pred_prob)}
        weights[f"Fold {fold+1}"] = pd.Series(model.coef_[0], model.feature_names_in_)
    results = pd.DataFrame(results).T
    weights = pd.DataFrame(weights)

    mask = (weights == 0).all(axis=1)
    exclude_feats = mask.index[mask].tolist()
    keep_feats = mask.index[~mask].tolist()
    print(
        f"{len(exclude_feats)} features with zero weights for all folds: {exclude_feats}"
    )
    print(
        f"{len(keep_feats)} features with non-zero weights in at least one fold: {keep_feats}"
    )
    cols = X.columns[X.columns.isin(keep_feats)]
    return cols
