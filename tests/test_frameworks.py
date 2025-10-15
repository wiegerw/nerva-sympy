#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import unittest
import numpy as np
from mlp_constructors import construct_models


def run_sgd_step(M, loss, X, T, lr):
    Y = M.feedforward(X)
    DY = loss.gradient(Y, T) / X.shape[0]
    M.backpropagate(Y, DY)
    M.optimize(lr)
    return Y, DY


def compute_loss(M, data_loader, loss):
    """Compute mean loss for a model over a data loader using the given loss."""
    N = len(data_loader.dataset)  # N is the number of examples
    total_loss = 0.0
    for X, T in data_loader:
        Y = M.feedforward(X)
        total_loss += loss(Y, T)
    return total_loss / N


def compute_accuracy(M, data_loader):
    import tensorflow as tf
    """Compute mean classification accuracy for a model over a data loader."""
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        Y = M.feedforward(X)
        predicted = tf.argmax(Y, axis=1)  # the predicted classes for the batch
        targets = tf.argmax(T, axis=1)    # the expected classes
        total_correct += tf.reduce_sum(tf.cast(tf.equal(predicted, targets), dtype=tf.int32))
    return total_correct / N


def tensors_to_numpy(t):
    if hasattr(t, "numpy"):
        return t.numpy()
    else:  # JAX or NumPy
        return np.array(t)


def max_deviation(arrays):
    base = arrays[0]
    return max(np.max(np.abs(base - a)) for a in arrays[1:])


class TestFrameworkConsistency(unittest.TestCase):
    """Unit test to check that all frameworks produce consistent intermediate results."""

    def test_all_frameworks(self):
        """Check that all frameworks produce consistent intermediates over a few batches."""
        n_steps = 50
        tol = 1e-5
        lr = 0.001

        here = os.path.dirname(__file__)
        data_path = os.path.join(here, "..", "data", "mnist-flattened.npz")
        models = construct_models(synchronize_weights=True, data_path=data_path)

        # unpack triples
        models_losses_loaders = list(models.values())

        for step, batches in enumerate(zip(*[ml[2] for ml in models_losses_loaders])):
            if step >= n_steps:
                break

            Xs, Ys, DYs, Ts = [], [], [], []
            for (M, loss, _), (X, T) in zip(models_losses_loaders, batches):
                Y, DY = run_sgd_step(M, loss, X, T, lr)
                Xs.append(tensors_to_numpy(X))
                Ys.append(tensors_to_numpy(Y))
                DYs.append(tensors_to_numpy(DY))
                Ts.append(tensors_to_numpy(T))

            self.assertLessEqual(max_deviation(Xs), tol, f"X mismatch at step {step}")
            self.assertLessEqual(max_deviation(Ys), tol, f"Y mismatch at step {step}")
            self.assertLessEqual(max_deviation(DYs), tol, f"DY mismatch at step {step}")
            self.assertLessEqual(max_deviation(Ts), tol, f"T mismatch at step {step}")


if __name__ == "__main__":
    unittest.main()
