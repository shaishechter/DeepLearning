from typing import Union

import numpy as np
import numba as nb

# activation functions
@nb.njit(parallel=True)
def sigmoid(z: Union[np.ndarray, float]):
    return 1 / (1 + np.exp(-z))


@nb.njit(parallel=True)
def tanh(z: Union[np.ndarray, float]):
    exp2z = np.exp(2 * z)
    return (exp2z**2 - 1) / (exp2z**2 + 1)


@nb.njit(parallel=True)
def relu(z: Union[np.ndarray, float]):
    return z * (z > 0)


def leaky_relu(z: Union[np.ndarray, float], neg_slope=0.01):
    return np.where(z > 0, z, neg_slope * z)

# cost functions
@nb.njit(parallel=True)
def CE_cost(y_hat: np.ndarray, y: np.ndarray):
    # cost_comps = - np.where(y == 1,
    #                         np.log(y_hat),
    #                         np.log(1-y_hat))
    cost_comps = - (y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
    return - cost_comps.sum() / cost_comps.shape[0]


@nb.njit(parallel=True)
def dw_sigmoid(X: np.ndarray, y_hat: np.ndarray, y: np.ndarray):
    return np.sum(
        (y_hat - y).reshape(-1, 1) * X,
        axis=0,
    ) / y.shape[0]



@nb.njit(parallel=True)
def dw_relu(X: np.ndarray, y_hat: np.ndarray, y: np.ndarray):
    return np.sum(
        (
                (y / y_hat) - ((1-y) / (1-y_hat)) * (y_hat > 0)
         ).reshape(-1, 1) * X,
        axis=0,
    ) / y.shape[0]
