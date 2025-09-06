#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import unittest
import numpy as np
from mlp_constructors import construct_models


def run_sgd_step(M, loss, X, T):
    Y = M.feedforward(X)
    DY = loss.gradient(Y, T) / X.shape[0]
    M.backpropagate(Y, DY)
    return Y, DY


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

    @classmethod
    def setUpClass(cls):
        """Set up the models once for all tests."""
        here = os.path.dirname(__file__)
        data_path = os.path.join(here, "..", "data", "mnist-flattened.npz")
        cls.models = construct_models(synchronize_weights=True, data_path=data_path)

    def test_all_frameworks(self):
        """Check that all frameworks produce consistent intermediates over a few batches."""
        n_steps = 3
        tol = 1e-6

        # unpack triples
        models_losses_loaders = list(self.models.values())

        for step, batches in enumerate(zip(*[ml[2] for ml in models_losses_loaders])):
            if step >= n_steps:
                break

            Xs, Ys, DYs, Ts = [], [], [], []
            for (M, loss, _), (X, T) in zip(models_losses_loaders, batches):
                Y, DY = run_sgd_step(M, loss, X, T)
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


if __name__ == "__main__":
    unittest.main()
