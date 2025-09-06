#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""
Interactive validation script:
Runs four frameworks (JAX, NumPy, TensorFlow, PyTorch), collects intermediate
deviations (X, Y, DY) over a number of batches, and plots them.
"""

import os
import matplotlib.pyplot as plt
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


def validate_frameworks_plot(models: dict, n_steps: int = 20):
    """
    Runs frameworks, collects deviations, and plots them.

    Args:
        models: dict mapping framework name -> (M, loss, loader)
        n_steps: number of training steps (batches) to compare
    """
    names = list(models.keys())
    models_losses_loaders = list(models.values())
    history = {name: {"X": [], "Y": [], "DY": []} for name in names}

    for step, batches in enumerate(zip(*[ml[2] for ml in models_losses_loaders])):
        if step >= n_steps:
            break

        Xs, Ys, DYs = [], [], []
        for (M, loss, _), (X, T) in zip(models_losses_loaders, batches):
            Y, DY = run_sgd_step(M, loss, X, T)
            Xs.append(tensors_to_numpy(X))
            Ys.append(tensors_to_numpy(Y))
            DYs.append(tensors_to_numpy(DY))

        # measure deviations relative to the first framework as reference
        for i, name in enumerate(names):
            history[name]["X"].append(np.max(np.abs(Xs[i] - Xs[0])))
            history[name]["Y"].append(np.max(np.abs(Ys[i] - Ys[0])))
            history[name]["DY"].append(np.max(np.abs(DYs[i] - DYs[0])))

    # Plot deviations
    plt.figure(figsize=(12, 6))
    for name in names:
        plt.plot(history[name]["X"], label=f"{name} X")
        plt.plot(history[name]["Y"], label=f"{name} Y")
        plt.plot(history[name]["DY"], label=f"{name} DY")

    plt.xlabel("Batch step")
    plt.ylabel("Max deviation from reference (first framework)")
    plt.title("Intermediate deviations over time")
    plt.yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    data_path = os.path.join(here, "..", "data", "mnist-flattened.npz")
    models = construct_models(synchronize_weights=True, data_path=data_path)

    # Number of batches to evaluate
    n_steps = 600

    # Run validation and plot deviations
    validate_frameworks_plot(models, n_steps=n_steps)
