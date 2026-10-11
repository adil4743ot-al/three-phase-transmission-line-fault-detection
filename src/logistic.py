
import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def add_intercept(X):
    return np.column_stack((np.ones(X.shape[0]), X))


def hypothesis(X, theta):
    return sigmoid(X @ theta)


def compute_cost(X, y, theta):
    m = len(y)

    h = hypothesis(X, theta)
    h = np.clip(h, 1e-12, 1 - 1e-12)

    cost = -(1 / m) * np.sum(
        y * np.log(h) + (1 - y) * np.log(1 - h)
    )

    return cost


def fit_logistic(X, y, alpha=0.1, iterations=2000):
    m, n = X.shape

    theta = np.zeros(n)
    cost_history = []

    for i in range(iterations):
        h = hypothesis(X, theta)

        gradient = (X.T @ (h - y)) / m

        theta = theta - alpha * gradient

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, np.array(cost_history)


def predict_proba(X, theta):
    return hypothesis(X, theta)


def predict(X, theta):
    probabilities = predict_proba(X, theta)
    return (probabilities >= 0.5).astype(int)
