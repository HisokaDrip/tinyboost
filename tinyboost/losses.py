import numpy as np


class SquaredError:
    name = "l2"

    def init_score(self, y):
        return float(np.mean(y))

    def grad_hess(self, y, raw):
        return raw - y, np.ones_like(raw)

    def loss(self, y, raw):
        return float(np.mean((y - raw) ** 2))

    def transform(self, raw):
        return raw


class LogLoss:
    name = "logloss"

    def init_score(self, y):
        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    def grad_hess(self, y, raw):
        p = sigmoid(raw)
        # floor the hessian so a confident leaf can't blow up to +-inf
        return p - y, np.maximum(p * (1 - p), 1e-16)

    def loss(self, y, raw):
        # log(1 + e^raw) - y*raw, written so it doesn't overflow
        return float(np.mean(np.logaddexp(0, raw) - y * raw))

    def transform(self, raw):
        return sigmoid(raw)


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))  # same thing, but never overflows
