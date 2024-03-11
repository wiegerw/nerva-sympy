#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from nerva_sympy.activation_functions import *
from nerva_sympy.matrix_operations import *

from utilities import equal_matrices, matrix, squared_error


class TestLinearLayers(TestCase):

    def test_linear_layer(self):
        D = 3
        K = 2
        N = 2
        loss = squared_error

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        X = x
        W = w

        # feedforward
        Y = X * W.T + row_repeat(b, N)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DW = DY.T * X
        Db = columns_sum(DY)
        DX = DY * W

        # test gradients
        DW1 = gradient(loss(Y), w)
        Db1 = gradient(loss(Y), b)
        DX1 = gradient(loss(Y), x)

        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_activation_layer(self):
        D = 3
        K = 2
        N = 2
        loss = squared_error
        act = HyperbolicTangentActivation()

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        X = x
        W = w

        # feedforward
        Z = X * W.T + row_repeat(b, N)
        Y = act(Z)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(DY, act.gradient(Z))
        DW = DZ.T * X
        Db = columns_sum(DZ)
        DX = DZ * W

        # test gradients
        DZ1 = substitute(gradient(loss(act(z)), z), (z, Z))
        DW1 = gradient(loss(Y), w)
        Db1 = gradient(loss(Y), b)
        DX1 = gradient(loss(Y), x)

        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1))
        self.assertTrue(equal_matrices(Db, Db1))
        self.assertTrue(equal_matrices(DX, DX1))

    def test_sigmoid_layer(self):
        D = 3
        K = 2
        N = 2
        loss = squared_error
        sigma = Sigmoid

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        X = x
        W = w

        # feedforward
        Z = X * W.T + row_repeat(b, N)
        Y = Sigmoid(Z)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(DY, hadamard(Y, ones(N, K) - Y))
        DW = DZ.T * X
        Db = columns_sum(DZ)
        DX = DZ * W

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


if __name__ == '__main__':
    import unittest
    unittest.main()