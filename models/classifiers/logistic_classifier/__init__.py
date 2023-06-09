"""Implementation of a logistic classifier using gradient descent"""
import warnings
from typing import Optional, Callable

import numba as nb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.exceptions import NotFittedError

from models.classifiers.base import BaseGDEstimator
from models.classifiers.utils import sigmoid, CE_cost as cost, dw_sigmoid as dw


class LogisticClassifier(BaseGDEstimator):
    def __init__(self,
                 learning_rate: float = 0.1,
                 max_iterations: int = 10_000,
                 stopping_criterion: float = 1e-7):
        super().__init__(learning_rate, max_iterations, stopping_criterion)
        self.weights: Optional[np.ndarray] = None
        self.activation_func: Callable = sigmoid

    @staticmethod
    @nb.njit
    def _fit(X_bar, y, weights, learning_rate, max_iterations, stopping_criterion):
        c = np.inf
        for _ in range(max_iterations):
            y_hat = sigmoid(
                (weights.reshape(1, -1) * X_bar).sum(axis=1)
            )
            weights = (
                    weights -
                    learning_rate * dw(X_bar, y_hat, y)
            )
            c_, c = c, cost(y_hat, y)
            if np.abs(c - c_) < stopping_criterion:
                converged = True
                break
        else:
            converged = False
        return weights, converged

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            initial_weights: Optional[np.ndarray] = None):
        self.weights = (
            initial_weights if initial_weights is not None
            else np.random.rand(X.shape[1] + 1)
        )
        X_bar = np.hstack((np.ones((X.shape[0], 1)), np.asarray(X)))
        y = np.asarray(y)
        self.weights, converged = LogisticClassifier._fit(
            X_bar,
            y,
            self.weights,
            self.learning_rate,
            self.max_iterations,
            self.stopping_criterion,
        )
        if not converged:
            warnings.warn(
                f"Fitting finished after {self.max_iterations} "
                f"iterations, convergence unlikely"
            )
        return self

    @staticmethod
    @nb.njit
    def _predict(X_bar: np.ndarray,
                 weights: np.ndarray) -> np.ndarray:
        return (weights.reshape(1, -1) * X_bar).sum(axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            raise NotFittedError
        if self.weights.shape[0] != X.shape[1] + 1:
            raise ValueError('dimension mismatch, the number of columns for X '
                             f'must be {self.weights.shape[0]}')
        X_bar = np.hstack((np.ones((X.shape[0], 1)), np.asarray(X)))
        # X_bar is of the form [1, X], and the first value of weights is assumed
        # to be the bias
        return self.activation_func(
            self._predict(X_bar, self.weights)
        )


if __name__=="__main__":
    model = LogisticClassifier(max_iterations=2000)
    n_obs = 10_000
    m_features = 20
    X, y = np.random.rand(n_obs, m_features), np.random.choice([0, 1], n_obs)
    model.fit(X, y)
    print(confusion_matrix(y, model.predict(X) > 0.5))
