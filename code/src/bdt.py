import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from ucimlrepo import fetch_ucirepo
from itertools import chain, zip_longest
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def compute_prob_density(x, axis=0):
    mean, std = np.mean(np.array(x), axis=axis), np.std(np.array(x), axis=axis)
    return lambda x: np.exp(-0.5*((x - mean)/std)**2)/(std*np.sqrt(2*np.pi))


def euler(x0=0, dt=0, dy=0, steps=-1, thr=1e-3):
    steps = int(thr/dt) if steps < 1 else steps
    value = np.zeros_like(x0)
    for step in range(steps):
        x0 = x0 + dt
        value = value + dt*dy(x0)

    return value


def compute_probability(feature, thr=1e-3, integrate=True):
    if integrate:
        return lambda x: euler(x0=x, dy=compute_prob_density(feature), dt=1e-5, thr=thr)

    return lambda x: compute_prob_density(feature)(x)


class NaiveBayes:
    def __init__(self, integrate=True):
        self.integrate = integrate
        self.classes = None
        self.indices = None
        self.class_conditionals = None
        self.class_probabilities = None
        self._probabilities = None
        self._fitted = False
        self._prob_dist = {}

    @property
    def probabilities(self):
        prob = pd.concat(self._probabilities, axis=1)
        prob.columns = self.classes
        return prob

    def _class_conditionals(self, x, c):
        func = self._prob_dist[c]
        class_conditionals = np.prod(func(x), axis=1) * self.class_probabilities[c]
        class_conditionals /= np.sum(
            class_conditionals)  # compute conditional probability for each sample by dividing by the sum of samples
        return class_conditionals

    def predict(self, x):
        probabilities = []
        if not self._fitted:
            for i, (c, class_indices) in enumerate(zip(self.classes, self.indices)):
                # Create and fit a prob_dist function to this class
                self._prob_dist[c] = compute_probability(feature=x.iloc[class_indices], thr=1e-3,
                                                         integrate=self.integrate)
                probabilities.append(self._class_conditionals(x=x, c=c))  # store probabilities

            self._fitted = True
        else:
            for i, c in enumerate(self.classes):
                probabilities.append(self._class_conditionals(x=x, c=c))  # store probabilities

        self._probabilities = probabilities
        return [self.classes[_] for _ in
                np.argmax(pd.DataFrame(np.array(probabilities).transpose(), columns=self.classes), axis=1)]

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)

        if isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

        # Compute the prior for the class probabilities P(y)
        self.class_probabilities = y.value_counts() / len(y)

        # Extract the classes
        self.classes = list(chain(*self.class_probabilities.index))

        # Extract indices
        self.indices = list(y.groupby(list(y.columns)).groups.values())

        return self.predict(x=x)