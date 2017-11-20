import torch
import numpy as np


def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""
    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2*variance) - 0.5 * np.log(2 * np.pi) - log_std
    return log_density
