import torch
import torch.nn as nn
from .tool_utils import *
import numpy as np

def distance_log_hook(out_table, initial_params, initial_abs_params=None, log_folded=False, log_freq=5):
    def log_distance(mod, log_info):
        if log_info['epoch'] % log_freq == 1:
            curr_params = get_params(mod)
            dist = get_norm(curr_params - initial_params)
            dist_fold = 0
            if log_folded:
                curr_abs_params = get_abs_weights(mod)
                dist_fold = get_norm(curr_abs_params - initial_abs_params)
            param_norms = np.asarray([get_norm(initial_params), 
				      get_norm(curr_params)])

            log_info_custom = {'epoch': log_info['epoch'],
                               'weight_norm': param_norms,
                               'dist': dist,
                               'dist_fold': dist_fold}
            out_table.append_row(log_info_custom)
    return log_distance
