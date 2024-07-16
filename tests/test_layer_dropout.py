#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from nerva_sympy.activation_functions import *
from nerva_sympy.matrix_operations import *

from utilities import equal_matrices, matrix, squared_error


class TestDropoutLayers(TestCase):

    def _test_linear_dropout_layer(self, D, K, N, loss):
        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        r = matrix('r', D, K)
        X = x
        W = w
        R = r

        # feedforward
        Y = X * hadamard(W.T, R) + row_repeat(b, N)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DW = hadamard(DY.T * X, R.T)
        Db = columns_sum(DY)
        DX = DY * hadamard(W, R.T)

        # test gradients
        DW1 = gradient(loss(Y), w)
        Db1 = gradient(loss(Y), b)
        DX1 = gradient(loss(Y), x)

        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_linear_dropout_layer(self):
        for loss in [elements_sum, squared_error]:
            self._test_linear_dropout_layer(D=3, K=2, N=2, loss=loss)
            self._test_linear_dropout_layer(D=2, K=3, N=2, loss=loss)
            self._test_linear_dropout_layer(D=2, K=2, N=3, loss=loss)

    def _test_activation_dropout_layer(self, D, K, N, loss, act):
        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        r = matrix('r', D, K)
        X = x
        W = w
        R = r

        # feedforward
        Z = X * hadamard(W.T, R) + row_repeat(b, N)
        Y = act(Z)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(DY, act.gradient(Z))
        DW = hadamard(DZ.T * X, R.T)
        Db = columns_sum(DZ)
        DX = DZ * hadamard(W, R.T)

        # test gradients
        DZ1 = substitute(gradient(loss(act(z)), z), (z, Z))
        DW1 = gradient(loss(Y), w)
        Db1 = gradient(loss(Y), b)
        DX1 = gradient(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_activation_dropout_layer(self):
        alpha = sp.symbols('alpha')
        for act in [LeakyReLUActivation(alpha), AllReLUActivation(alpha), HyperbolicTangentActivation(), SigmoidActivation()]:
            for loss in [elements_sum, squared_error]:
                self._test_activation_dropout_layer(D=3, K=2, N=2, loss=loss, act=act)
                self._test_activation_dropout_layer(D=2, K=3, N=2, loss=loss, act=act)
                self._test_activation_dropout_layer(D=2, K=2, N=3, loss=loss, act=act)
        # N.B. ReLUActivation() is excluded, since it causes problems in combination with squared_error.
        # This seems to be a technicality in SymPy, but it is unknown how to resolve it. For this specific
        # case a term evaluates to `nan == nan`, which returns False.

    def _test_sigmoid_dropout_layer(self, D, K, N, loss):
        sigma = Sigmoid

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        r = matrix('r', D, K)
        X = x
        W = w
        R = r

        # feedforward
        Z = X * hadamard(W.T, R) + row_repeat(b, N)
        Y = Sigmoid(Z)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(DY, hadamard(Y, ones(N, K) - Y))
        DW = hadamard(DZ.T * X, R.T)
        Db = columns_sum(DZ)
        DX = DZ * hadamard(W, R.T)

        # test gradients
        Y_z = Sigmoid(z)
        DZ1 = substitute(gradient(loss(Y_z), z), (z, Z))
        DW1 = gradient(loss(Y), w)
        Db1 = gradient(loss(Y), b)
        DX1 = gradient(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_sigmoid_dropout_layer(self):
        for loss in [elements_sum, squared_error]:
            self._test_sigmoid_dropout_layer(D=3, K=2, N=2, loss=loss)
            self._test_sigmoid_dropout_layer(D=2, K=3, N=2, loss=loss)
            self._test_sigmoid_dropout_layer(D=2, K=2, N=3, loss=loss)


if __name__ == '__main__':
    import unittest
    unittest.main()