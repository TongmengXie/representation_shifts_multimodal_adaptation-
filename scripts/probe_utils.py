"""Utilities for training and evaluating linear probes.

This module provides simple wrappers around scikit‑learn to train
logistic regression classifiers for probing tasks. Functions here
abstract away the boilerplate associated with fitting a classifier,
performing a train/test split, and computing metrics. They return
structured results suitable for further analysis or plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


@dataclass
class ProbeResult:
    """Dataclass to hold probe evaluation statistics."""
    train_accuracy: float
    test_accuracy: float
    test_f1: float
    classification_report: str


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
) -> ProbeResult:
    """Train a logistic regression probe and compute metrics.

    The data is split into train and test sets according to the
    ``test_size`` parameter with stratification. A logistic regression
    classifier is fitted to the training portion. Accuracy and macro
    F1 are reported on both the training and test sets. The full
    classification report (precision, recall, F1 per class) is also
    returned as a string for more detailed inspection.

    Parameters
    ----------
    features: np.ndarray
        Feature matrix of shape ``[N, D]`` where ``N`` is the number of
        examples and ``D`` is the dimensionality of the representation
        being probed.
    labels: np.ndarray
        Binary or multiclass labels of length ``N``.
    test_size: float, optional
        Fraction of the data used for the test split. Defaults to 0.2.
    random_state: int, optional
        Random seed for the train/test split and classifier. Defaults
        to 42.
    max_iter: int, optional
        Maximum number of iterations for the logistic regression solver.
        Defaults to 1000.

    Returns
    -------
    ProbeResult
        A dataclass containing train accuracy, test accuracy, test F1,
        and the detailed classification report.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    clf = LogisticRegression(max_iter=max_iter, random_state=random_state).fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average="macro")
    report = classification_report(y_test, y_pred_test)
    return ProbeResult(
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        test_f1=f1,
        classification_report=report,
    )


def train_linear_probe_on_splits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_iter: int = 1000,
    random_state: int = 42,
) -> ProbeResult:
    """Train a logistic regression probe on predefined train/test splits.

    Unlike :func:`train_linear_probe`, this function accepts explicit
    training and testing feature matrices and labels. This is useful
    for experiments where a fixed split (e.g., groupwise control) has
    already been created and should not be re-sampled internally.

    Parameters
    ----------
    X_train: np.ndarray
        Feature matrix for the training examples, shape ``[N_train, D]``.
    y_train: np.ndarray
        Labels for the training examples, length ``N_train``.
    X_test: np.ndarray
        Feature matrix for the test examples, shape ``[N_test, D]``.
    y_test: np.ndarray
        Labels for the test examples, length ``N_test``.
    max_iter: int, optional
        Maximum number of iterations for the logistic regression solver.
        Defaults to 1000.
    random_state: int, optional
        Random seed used to initialise the classifier weights.
        Defaults to 42.

    Returns
    -------
    ProbeResult
        Dataclass containing train accuracy, test accuracy, macro F1,
        and the detailed classification report.
    """
    clf = LogisticRegression(max_iter=max_iter, random_state=random_state).fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average="macro")
    report = classification_report(y_test, y_pred_test)
    return ProbeResult(
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        test_f1=f1,
        classification_report=report,
    )

