"""Learned scheduler selector.

For a window of pending requests, a small decision tree predicts which
classical scheduler will give the least head movement, together with a
confidence score (the fraction of training windows in the same tree leaf
that had that label).
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from features import extract
from schedulers import SCHEDULERS, LEFT, RIGHT

# When several schedulers tie on movement, the label goes to the first one in
# this list: simpler and fairer policies first, SSTF last (starvation risk).
PREFERENCE = ["FCFS", "SCAN", "C-SCAN", "SSTF"]


def movements(requests, head, direction):
    return {name: f(requests, head, direction).movement for name, f in SCHEDULERS.items()}


def best_label(costs):
    lowest = min(costs.values())
    return next(name for name in PREFERENCE if costs[name] == lowest)


def make_dataset(rng, generators, n_per_type):
    """Random windows with random head position and direction, labelled by
    running all four schedulers on each window."""
    X, y, costs, kinds = [], [], [], []
    for kind, gen in generators.items():
        for _ in range(n_per_type):
            reqs = gen(rng)
            head = int(rng.integers(0, 200))
            direction = LEFT if rng.random() < 0.5 else RIGHT
            c = movements(reqs, head, direction)
            X.append(extract(reqs, head, direction))
            y.append(best_label(c))
            costs.append(c)
            kinds.append(kind)
    return np.array(X), np.array(y), costs, kinds


class Selector:
    def __init__(self, max_depth=4, min_samples_leaf=10, seed=0):
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed
        )

    def fit(self, X, y):
        self.tree.fit(X, y)
        return self

    def choose(self, requests, head, direction):
        """Return (scheduler name, confidence, feature vector)."""
        x = extract(requests, head, direction)
        proba = self.tree.predict_proba([x])[0]
        k = int(np.argmax(proba))
        return self.tree.classes_[k], float(proba[k]), x
