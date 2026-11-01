#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# Must be called before importing any frameworks
import utilities
utilities.require_backends('jax', 'numpy', 'tensorflow', 'torch')
utilities.disable_gpu()

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


def tensors_to_numpy(t):
    if hasattr(t, "numpy"):
        return t.numpy()
    else:  # JAX or NumPy
        return np.array(t)


def max_deviation(arrays, atol=6e-5, rtol=1e-6):
    # Cast all arrays to float32 to ensure unified dtype for comparisons
    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    base = arrays[0]
    # Compute per-array max deviation using allclose-like criterion
    max_dev = 0.0
    for a in arrays[1:]:
        diff = np.abs(base - a)
        # Effective tolerance per element: atol + rtol * |base|
        tol = atol + rtol * np.abs(base)
        # Only consider the amount beyond tolerance, clipped at 0
        excess = np.maximum(diff - tol, 0.0)
        if excess.size:
            max_dev = max(max_dev, float(np.max(excess)))
    return max_dev


class TestFrameworkConsistency(unittest.TestCase):
    """Unit test to check that all frameworks produce consistent intermediate results."""

    def test_all_frameworks(self):
        """Check that all frameworks produce consistent intermediates over a few batches."""
        n_steps = 50
        tol = 1e-5
        lr = 0.001

        # Use a fixed seed by default, so that failures are reproducible.
        # Set NERVA_TEST_SEED to use a different seed.
        seed = int(os.environ.get("NERVA_TEST_SEED", "12345"))
        print(f"[nerva-tests] Using seed: {seed}")
        utilities.set_test_seeds(seed)

        here = os.path.dirname(__file__)
        data_path = os.path.join(here, "..", "..", "data", "mnist-flattened.npz")
        if not os.path.exists(data_path):
            self.skipTest(f"Dataset '{data_path}' not found. Create it with: cd data && python prepare_data.py --dataset=mnist --download")
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

            self.assertLessEqual(max_deviation(Xs), tol, f"X mismatch at step {step} (seed={seed})")
            self.assertLessEqual(max_deviation(Ys), tol, f"Y mismatch at step {step} (seed={seed})")
            self.assertLessEqual(max_deviation(DYs), tol, f"DY mismatch at step {step} (seed={seed})")
            self.assertLessEqual(max_deviation(Ts), tol, f"T mismatch at step {step} (seed={seed})")


if __name__ == "__main__":
    unittest.main()
