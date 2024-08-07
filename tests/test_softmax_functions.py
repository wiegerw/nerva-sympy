#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

import nerva_jax.softmax_functions as jnp_
import nerva_numpy.softmax_functions as np_
import nerva_sympy.softmax_functions as sympy_
import nerva_tensorflow.softmax_functions as tf_
import nerva_torch.softmax_functions as torch_
import numpy as np
from nerva_sympy.matrix_operations import *
from nerva_sympy.softmax_functions import *

from utilities import check_arrays_equal, check_numbers_equal, to_jax, to_numpy, to_sympy, to_tensorflow, to_torch

Matrix = sp.Matrix

#-------------------------------------#
# alternative implementations of softmax functions
#-------------------------------------#

def softmax1(X: Matrix) -> Matrix:
    N, D = X.shape

    def softmax(x):
        e = exp(x)
        return e / sum(e)

    return join_rows([softmax(X.row(i)) for i in range(N)])


class TestSoftmax(TestCase):
    def test_softmax(self):
        D = 3
        N = 2
        X = Matrix(sp.symarray('x', (N, D), real=True))

        y1 = softmax(X)
        y2 = softmax1(X)
        y3 = stable_softmax(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(N, D))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(N, D))

        y1 = log_softmax(X)
        y2 = log(softmax(X))
        y3 = stable_log_softmax(X)
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(N, D))
        self.assertEqual(sp.simplify(y1 - y3), sp.zeros(N, D))

    def test_softmax_jacobian(self):
        x = Matrix(sp.symbols('x y z'), real=True).T
        N, D = x.shape

        y1 = sp.simplify(softmax_jacobian(x))
        y2 = sp.simplify(jacobian(softmax(x), x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, D))

    def test_log_softmax_jacobian(self):
        x = Matrix(sp.symbols('x y z'), real=True).T
        N, D = x.shape

        y1 = sp.simplify(log_softmax_jacobian(x))
        y2 = sp.simplify(jacobian(log_softmax(x), x))
        self.assertEqual(sp.simplify(y1 - y2), sp.zeros(D, D))


class TestSoftmaxValues(TestCase):
    def make_variables(self):
        X = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=np.float32)

        x = np.array([
            [11, 2, 3]
        ], dtype=np.float32)

        return X, x

    def _test_softmax(self, function_name, x):
        f_sympy = getattr(sympy_, function_name)
        f_numpy = getattr(np_, function_name)
        f_tensorflow = getattr(tf_, function_name)
        f_torch = getattr(torch_, function_name)
        f_jax = getattr(jnp_, function_name)

        x1 = f_sympy(to_sympy(x))
        x2 = f_numpy(to_numpy(x))
        x3 = f_tensorflow(to_tensorflow(x))
        x4 = f_torch(to_torch(x))
        x5 = f_jax(to_jax(x))

        if isinstance(x1, sp.Matrix):
            check_arrays_equal(self, function_name, [x1, x2, x3, x4, x5])
        else:
            check_numbers_equal(self, function_name, [x1, x2, x3, x4, x5])

    def test_all(self):
        X, x = self.make_variables()
        self._test_softmax('softmax', X)
        self._test_softmax('softmax_jacobian', x)
        self._test_softmax('stable_softmax', X)
        self._test_softmax('log_softmax', X)
        self._test_softmax('log_softmax_jacobian', x)
        self._test_softmax('stable_log_softmax', X)


if __name__ == '__main__':
    import unittest
    unittest.main()