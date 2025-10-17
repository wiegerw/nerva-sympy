# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
from typing import List, Tuple, Union
from unittest import TestCase

import jax.numpy as jnp
import numpy as np
import sympy as sp
from sympy import Matrix
import tensorflow as tf
import torch

def disable_gpu():
    """Force CPU-only execution across common frameworks."""
    # Must be called before importing any frameworks
    import nerva_jax.utilities
    import nerva_tensorflow.utilities
    import nerva_torch.utilities
    nerva_jax.utilities.disable_gpu()
    nerva_tensorflow.utilities.disable_gpu()
    nerva_torch.utilities.disable_gpu()


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


def to_numpy(x: Union[sp.Matrix, np.ndarray, torch.Tensor, tf.Tensor, tf.Variable, jnp.ndarray]) -> np.ndarray:
    if isinstance(x, sp.Matrix):
        return np.array(x.tolist(), dtype=np.float64)
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, tf.Tensor):
        return x.numpy()
    elif isinstance(x, tf.Variable):
        return x.numpy()
    elif isinstance(x, jnp.ndarray):
        return np.array(x)
    else:
        raise ValueError("Unsupported input type. Input must be one of sp.Matrix, np.ndarray, torch.Tensor, or tf.Tensor.")


def to_sympy(X: np.ndarray) -> sp.Matrix:
    return sp.Matrix(X)


def to_torch(X: np.ndarray) -> torch.Tensor:
    return torch.Tensor(X)


def to_tensorflow(X: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(X)


def to_jax(X: np.ndarray) -> jnp.ndarray:
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
