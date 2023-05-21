from collections.abc import Callable
from typing import Optional

from flatbuffers.builder import np
from sklearn.base import BaseEstimator

from models.classifiers.utils import sigmoid


class BaseGDEstimator(BaseEstimator):
    def __init__(self,
                 learning_rate: float = 0.1,
                 max_iterations: int = 10_000,
                 stopping_criterion: float = 1e-7):
        """
        Base class for all estimators whose fitting is based on gradient descent.
        :param learning_rate: constant by which the gradient is multiplied when
            descending.
        :param max_iterations: maximum number of iterations of GD until the
            fitting process halts.
        :param stopping_criterion: once the change in cost between consecutive
            descent steps falls below this value, the GD algorithm halts.
        """
        self.learning_rate: float = learning_rate
        self.max_iterations = max_iterations
        self.stopping_criterion = stopping_criterion