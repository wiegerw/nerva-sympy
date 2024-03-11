#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import random
from unittest import TestCase

import nerva_jax.loss_functions as jnp_
import nerva_numpy.loss_functions as np_
import nerva_sympy.loss_functions as sympy_
import nerva_tensorflow.loss_functions as tf_
import nerva_torch.loss_functions as torch_
import numpy as np
import sympy as sp
from nerva_sympy.matrix_operations import gradient, substitute

from utilities import check_arrays_equal, check_numbers_equal, equal_matrices, matrix, to_jax, to_numpy, to_sympy, \
    to_tensorflow, to_torch


def instantiate_one_hot(X: sp.Matrix) -> sp.Matrix:
    m, n = X.shape
    X0 = sp.zeros(m, n)
    for i in range(m):
        j = random.randrange(0, n)
        X0[i, j] = 1

    return X0


class TestLossFunctionGradients(TestCase):
    def make_variables(self):
        K = 3
        N = 2
        y = matrix('y', 1, K)
        t = matrix('t', 1, K)
        Y = matrix('Y', N, K)
        T = matrix('T', N, K)
        return K, y, t, Y, T

    def _test_loss_function(self, function_name: str):
        K, y, t, Y, T = self.make_variables()

        # retrieve functions by name
        loss_value = getattr(sympy_, function_name)
        loss_gradient = getattr(sympy_, f'{function_name}_gradient')
        Loss_value = getattr(sympy_, function_name.capitalize())
        Loss_gradient = getattr(sympy_, f'{function_name.capitalize()}_gradient')

        loss = loss_value(y, t)
        Dy1 = loss_gradient(y, t)
        Dy2 = gradient(loss, y)
        self.assertTrue(equal_matrices(Dy1, Dy2))

        loss = Loss_value(Y, T)
        DY1 = Loss_gradient(Y, T)
        DY2 = gradient(loss, Y)
        self.assertTrue(equal_matrices(DY1, DY2))

    def _test_loss_function_one_hot(self, function_name: str):
        K, y, t, Y, T = self.make_variables()

        # retrieve functions by name
        loss_gradient = getattr(sympy_, f'{function_name}_gradient')
        Loss_gradient = getattr(sympy_, f'{function_name.capitalize()}_gradient')
        loss_gradient_one_hot = getattr(sympy_, f'{function_name}_gradient_one_hot')
        Loss_gradient_one_hot = getattr(sympy_, f'{function_name.capitalize()}_gradient_one_hot')

        Dy = loss_gradient(y, t)
        DY = Loss_gradient(Y, T)

        # test with a one-hot encoded vector t0
        t_0 = instantiate_one_hot(t)
        Dy1 = substitute(loss_gradient_one_hot(y, t), (t, t_0))
        Dy2 = substitute(Dy, (t, t_0))
        self.assertTrue(equal_matrices(Dy1, Dy2))

        # test with a one-hot encoded matrix T0
        T0 = instantiate_one_hot(T)
        DY1 = substitute(Loss_gradient_one_hot(Y, T), (T, T0))
        DY2 = substitute(DY, (T, T0))
        self.assertTrue(equal_matrices(DY1, DY2))

    def test_squared_error_loss(self):
        self._test_loss_function('squared_error_loss')

    def test_mean_squared_error_loss(self):
        self._test_loss_function('mean_squared_error_loss')

    def test_cross_entropy_loss(self):
        self._test_loss_function('cross_entropy_loss')

    def test_softmax_cross_entropy_loss(self):
        self._test_loss_function('softmax_cross_entropy_loss')
        self._test_loss_function_one_hot('softmax_cross_entropy_loss')

    def test_stable_softmax_cross_entropy_loss(self):
        self._test_loss_function('stable_softmax_cross_entropy_loss')
        self._test_loss_function_one_hot('stable_softmax_cross_entropy_loss')

    def test_logistic_cross_entropy_loss(self):
        self._test_loss_function('logistic_cross_entropy_loss')

    def test_negative_log_likelihood_loss(self):
        self._test_loss_function('negative_log_likelihood_loss')


class TestLossFunctionValues(TestCase):
    def make_variables(self):
        y = np.array([
            [11, 2, 3]
        ], dtype=np.float32)

        t = np.array([
            [0, 1, 0]
        ], dtype=np.float32)

        Y = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=np.float32)

        T = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)

        return y, t, Y, T

    def _test_loss_function(self, function_name: str):
        y, t, Y, T = self.make_variables()

        print('=== test loss on vectors ===')
        name = function_name
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        f_jax = getattr(jnp_, name)
        x1 = f_sympy(to_sympy(y), to_sympy(t))
        x2 = f_numpy(to_numpy(y), to_numpy(t))
        x3 = f_tensorflow(to_tensorflow(y), to_tensorflow(t))
        x4 = f_torch(to_torch(y), to_torch(t))
        x5 = f_jax(to_jax(y), to_jax(t))
        check_numbers_equal(self, function_name, [x1, x2, x3, x4, x5])

        print('=== test loss gradient on vectors ===')
        name = f'{function_name}_gradient'
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        f_jax = getattr(jnp_, name)
        x1 = f_sympy(to_sympy(y), to_sympy(t))
        x2 = f_numpy(to_numpy(y), to_numpy(t))
        x3 = f_tensorflow(to_tensorflow(y), to_tensorflow(t))
        x4 = f_torch(to_torch(y), to_torch(t))
        x5 = f_jax(to_jax(y), to_jax(t))
        check_arrays_equal(self, function_name, [x1, x2, x3, x4, x5])

        print('=== test loss on matrices ===')
        name = function_name.capitalize()
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        f_jax = getattr(jnp_, name)
        x1 = f_sympy(to_sympy(Y), to_sympy(T))
        x2 = f_numpy(to_numpy(Y), to_numpy(T))
        x3 = f_tensorflow(to_tensorflow(Y), to_tensorflow(T))
        x4 = f_torch(to_torch(Y), to_torch(T))
        x5 = f_jax(to_jax(Y), to_jax(T))
        check_numbers_equal(self, function_name, [x1, x2, x3, x4, x5])

        print('=== test loss gradient on matrices ===')
        name = f'{function_name.capitalize()}_gradient'
        f_sympy = getattr(sympy_, name)
        f_numpy = getattr(np_, name)
        f_tensorflow = getattr(tf_, name)
        f_torch = getattr(torch_, name)
        f_jax = getattr(jnp_, name)
        x1 = f_sympy(to_sympy(Y), to_sympy(T))
        x2 = f_numpy(to_numpy(Y), to_numpy(T))
        x3 = f_tensorflow(to_tensorflow(Y), to_tensorflow(T))
        x4 = f_torch(to_torch(Y), to_torch(T))
        x5 = f_jax(to_jax(Y), to_jax(T))
        check_arrays_equal(self, function_name, [x1, x2, x3, x4, x5])

    def test_squared_error_loss(self):
        self._test_loss_function('squared_error_loss')

    def test_mean_squared_error_loss(self):
        self._test_loss_function('mean_squared_error_loss')

    def test_cross_entropy_loss(self):
        self._test_loss_function('cross_entropy_loss')

    def test_softmax_cross_entropy_loss(self):
        self._test_loss_function('softmax_cross_entropy_loss')

    def test_stable_softmax_cross_entropy_loss(self):
        self._test_loss_function('stable_softmax_cross_entropy_loss')

    def test_logistic_cross_entropy_loss(self):
        self._test_loss_function('logistic_cross_entropy_loss')

    def test_negative_log_likelihood_loss(self):
        self._test_loss_function('negative_log_likelihood_loss')


if __name__ == '__main__':
    import unittest
    unittest.main()