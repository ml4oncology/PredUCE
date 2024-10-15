from typing import Sequence
import os

import numpy as np


###############################################################################
# I/O
###############################################################################
def initialize_folders():
    main_folders = ["logs", "models", "result/tables"]
    for folder in main_folders:
        if not os.path.exists(f"./{folder}"):
            os.makedirs(f"./{folder}")


###############################################################################
# Thresholding
###############################################################################
def compute_threshold(pred: Sequence[float], desired_alarm_rate: float):
    """Compute the prediction threshold based on desired alarm rate"""
    for pred_threshold in np.arange(0, 1.0, 0.001):
        alarm_rate = np.mean(pred > pred_threshold)
        if np.isclose(alarm_rate, desired_alarm_rate, atol=0.005):
            return pred_threshold, alarm_rate
