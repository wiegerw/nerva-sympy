# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import importlib.util
import os
import sys
from typing import List, Tuple, Union
from unittest import SkipTest, TestCase
import random

import numpy as np
import sympy as sp
from sympy import Matrix

# The numerical frameworks (JAX, TensorFlow, PyTorch) are optional. They are imported lazily,
# so that the purely symbolic tests can run with only numpy and sympy installed.

# Maps a nerva backend name to the Python modules it needs.
BACKEND_MODULES = {
    'numpy': ['nerva_numpy', 'numpy'],
    'jax': ['nerva_jax', 'jax'],
    'tensorflow': ['nerva_tensorflow', 'tensorflow'],
    'torch': ['nerva_torch', 'torch'],
}


def require_modules(*names: str) -> None:
    """Skip the calling test module if any of the given modules is not installed.

    Call this at module level, before importing the modules. Both pytest and
    unittest report a module-level SkipTest as a skipped module.
    """
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise SkipTest(f"requires {', '.join(missing)} (install with: pip install 'nerva-sympy[all]')")


def slow(test):
    """Mark a test as slow. Slow tests are skipped with pytest -m "not slow".

    Without pytest installed (unittest), slow tests are run as usual.
    """
    try:
        import pytest
    except ImportError:
        return test
    return pytest.mark.slow(test)


def require_backends(*backends: str) -> None:
    """Skip the calling test module unless the given nerva backends and their frameworks are installed."""
    require_modules(*[name for backend in backends for name in BACKEND_MODULES[backend]])

def disable_gpu():
    """Force CPU-only execution across common frameworks."""
    # Must be called before importing any frameworks. Backends that are not installed are ignored.
    for backend in ('jax', 'tensorflow', 'torch'):
        if all(importlib.util.find_spec(name) is not None for name in BACKEND_MODULES[backend]):
            module = importlib.import_module(f'nerva_{backend}.utilities')
            module.disable_gpu()


def to_float(x):
    if hasattr(x, 'item'):
        # Handles numpy, torch, tensorflow tensors
        return float(x.item())
    elif hasattr(x, '__float__'):
        # Handles sympy.Float and other float-like types
        return float(x)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


VERBOSE = os.environ.get('NERVA_TEST_VERBOSE', '0') in ('1', 'true', 'True')


def set_test_seeds(seed: int):
    """Seed Python, NumPy, TensorFlow, and PyTorch RNGs.

    JAX is not seeded here because the test constructors use NumPy for init and
    JAX ops are deterministic given the inputs and parameters. If JAX random
    functions are introduced, seed them explicitly in the constructor.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def check_arrays_equal(testcase: TestCase, operation, values):
    if VERBOSE:
        print(f'--- {operation} ---')
    values = [to_numpy(x) for x in values]
    if VERBOSE:
        for x in values:
            print(x)
    x0 = values[0]
    for x in values[1:]:
        testcase.assertTrue(np.allclose(x0, x, atol=1e-5))


def check_numbers_equal(testcase: TestCase, operation, values):
    if VERBOSE:
        print(f'--- {operation} ---')
        for x in values:
            print(x, type(x))

    floats = [to_float(x) for x in values]
    x0 = floats[0]
    for x in floats[1:]:
        testcase.assertAlmostEqual(x0, x, delta=1e-5)


def to_numpy(x) -> np.ndarray:
    """Convert a sympy matrix, numpy array, or torch/tensorflow/jax tensor to a numpy array."""
    if isinstance(x, sp.Matrix):
        return np.array(x.tolist(), dtype=np.float64)
    elif isinstance(x, np.ndarray):
        return x
    # A tensor of a framework can only exist if that framework has been imported,
    # so checking sys.modules avoids importing frameworks that are not needed.
    torch = sys.modules.get('torch')
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    tf = sys.modules.get('tensorflow')
    if tf is not None and isinstance(x, (tf.Tensor, tf.Variable)):
        return x.numpy()
    jax = sys.modules.get('jax')
    if jax is not None and isinstance(x, jax.Array):
        return np.array(x)
    raise ValueError("Unsupported input type. Input must be one of sp.Matrix, np.ndarray, torch.Tensor, tf.Tensor or jax.Array.")


def to_sympy(X: np.ndarray) -> sp.Matrix:
    return sp.Matrix(X)


def to_torch(X: np.ndarray):
    import torch
    return torch.Tensor(X)


def to_tensorflow(X: np.ndarray):
    import tensorflow as tf
    return tf.convert_to_tensor(X)


def to_jax(X: np.ndarray):
    import jax.numpy as jnp
    return jnp.array(X)


def to_eigen(X: np.ndarray) -> np.ndarray:
    return np.asfortranarray(np.copy(X, order='C'))


def matrix(name: str, rows: int, columns: int) -> Matrix:
    return Matrix(sp.symarray(name, (rows, columns), real=True))


def to_matrix(x):
    return sp.Matrix([[x]])


def to_number(x: sp.Matrix):
    assert x.shape == (1, 1)
    return x[0, 0]


def column_vector(variables: List[str]) -> Matrix:
    symbols_list = sp.symbols(variables, real=True)
    return Matrix(symbols_list)


def row_vector(variables: List[str]) -> Matrix:
    symbols_list = sp.symbols(variables, real=True)
    return Matrix([symbols_list])


def substitute(expr, substitutions: Union[Tuple[Matrix, Matrix], List[Tuple[Matrix, Matrix]]]):
    if isinstance(substitutions, tuple):
        substitutions = [substitutions]
    for (X, Y) in substitutions:
        assert X.shape == Y.shape
        m, n = X.shape
        sigma = ((X[i, j], Y[i, j]) for i in range(m) for j in range(n))
        expr = expr.subs(sigma)
    return expr


def equal_matrices(A: Matrix, B: Matrix, simplify_arguments=False) -> bool:
    m, n = A.shape
    if simplify_arguments:
        A = sp.simplify(A)
        B = sp.simplify(B)
    return A.shape == B.shape and sp.simplify(A - B) == sp.zeros(m, n)


# A and B must depend on variables in X.
def equal_matrices_lite(A: Matrix, B: Matrix, X: Matrix, atol=1e-5) -> bool:
    rows, cols = X.shape
    X0 = np.random.rand(rows, cols)
    A0 = to_numpy(substitute(A, (X, Matrix(X0))))
    B0 = to_numpy(substitute(B, (X, Matrix(X0))))
    return np.allclose(A0, B0, atol=atol)


def instantiate(X: sp.Matrix, low=0, high=10) -> sp.Matrix:
    X0 = sp.Matrix(np.random.randint(low, high, X.shape))
    return X0


def squared_error(X: Matrix):
    m, n = X.shape

    def f(x: Matrix) -> float:
        return sp.sqrt(sum(xi * xi for xi in x))

    return sum(f(X.col(j)) for j in range(n))


def sum1(iterable, start=None):
    iterable = list(iterable)
    if not iterable:
        # nothing to sum → just return start or 0x0
        return start if start is not None else sp.zeros(0, 0)
    if start is None:
        first = iterable[0]
        if isinstance(first, sp.MatrixBase):
            start = sp.zeros(*first.shape)
        else:
            start = 0
    return sum(iterable, start)


def pp(name: str, x: sp.Matrix):
    if VERBOSE:
        print(f'{name} ({x.shape[0]}x{x.shape[1]})')
        for row in x.tolist():
            print('[', end='')
            for i, elem in enumerate(row):
                print(f'{elem}', end='')
                if i < len(row) - 1:
                    print(', ', end='')
            print(']')
        print()
