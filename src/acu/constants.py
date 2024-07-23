# model parameters
model_tuning_param = {
    "Ridge": {"C": (0.0001, 1)},
    "LASSO": {"C": (0.0001, 1)},
    "XGB": {
        "n_estimators": (50, 100),
        "max_depth": (3, 6),
        "learning_rate": (0.01, 0.3),
        "min_split_loss": (0, 0.5),
        "min_child_weight": (6, 100),
        "reg_lambda": (0, 1),  # L2 regularization
        "reg_alpha": (0, 1000),  # L1 regularization
    },
    "LGBM": {
        "n_estimators": (50, 200),
        "max_depth": (3, 6),
        "learning_rate": (0.01, 0.3),
        "num_leaves": (10, 40),
        "min_data_in_leaf": (10, 50),
        "feature_fraction": (0.5, 1),
        "bagging_fraction": (0.5, 1),
        "bagging_freq": (0, 10),
        "reg_lambda": (0, 1),
        "reg_alpha": (0, 1000),
    },
    "RF": {
        "n_estimators": (50, 100),
        "max_depth": (3, 6),
        "min_samples_leaf": (10, 50),
    },
    "SVC": {
        "C": (0.0001, 1),
        "kernel": (0, 3.99),
    },
}
bayesopt_param = {
    "Ridge": {"init_points": 2, "n_iter": 10},
    "LASSO": {"init_points": 2, "n_iter": 10},
    "XGB": {"init_points": 15, "n_iter": 200},
    "LGBM": {"init_points": 15, "n_iter": 200},
    "RF": {"init_points": 10, "n_iter": 50},
    "SVC": {"init_points": 10, "n_iter": 50},
}
model_static_param = {
    "Ridge": {
        "penalty": "l2",
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": 42,
    },
    "LASSO": {
        "penalty": "l1",
        "solver": "saga",
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": 42,
    },
    "XGB": {
        "random_state": 42,
    },
    "LGBM": {"random_state": 42, "verbosity": -1},
    "RF": {"random_state": 42},
    "SVC": {"random_state": 42, "probability": True},
}
