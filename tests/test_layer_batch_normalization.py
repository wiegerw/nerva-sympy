#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# see also https://docs.sympy.org/latest/modules/matrices/matrices.html

from unittest import TestCase

from nerva_sympy.matrix_operations import *

from utilities import equal_matrices, matrix, squared_error


class TestBatchNormalizationLayers(TestCase):

    def _test_simple_batch_normalization_layer(self, D, N, loss):
        K = D  # K and D are always equal in batch normalization

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        X = x

        # feedforward
        R = X - row_repeat(columns_mean(X), N)
        Sigma = diag(R.T * R).T / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Y = hadamard(row_repeat(inv_sqrt_Sigma, N), R)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DX = hadamard(row_repeat(inv_sqrt_Sigma / N, N), (N * identity(N) - ones(N, N)) * DY - hadamard(Y, row_repeat(diag(Y.T * DY).T, N)))

        # test gradients
        DX1 = gradient(loss(Y), x)

        self.assertTrue(equal_matrices(DX, DX1))

    def test_simple_batch_normalization_layer(self):
        self._test_simple_batch_normalization_layer(2, 3, loss=elements_sum)
        self._test_simple_batch_normalization_layer(3, 2, loss=elements_sum)

    def _test_affine_layer(self, D, N, loss):
        K = D

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        beta = matrix('beta', 1, K)
        gamma = matrix('gamma', 1, K)
        X = x

        # feedforward
        Y = hadamard(row_repeat(gamma, N), X) + row_repeat(beta, N)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DX = hadamard(row_repeat(gamma, N), DY)
        Dbeta = columns_sum(DY)
        Dgamma = columns_sum(hadamard(X, DY))

        # test gradients
        DX1 = gradient(loss(Y), x)
        Dbeta1 = gradient(loss(Y), beta)
        Dgamma1 = gradient(loss(Y), gamma)

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))

    def test_affine_layer(self):
        for loss in [elements_sum, squared_error]:
            self._test_affine_layer(D=3, N=2, loss=loss)
            self._test_affine_layer(D=2, N=3, loss=loss)

    def _test_batch_normalization_layer(self, D, N, loss):
        K = D  # K and D are always equal in batch normalization

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        beta = matrix('beta', 1, K)
        gamma = matrix('gamma', 1, K)
        X = x

        # feedforward
        R = X - row_repeat(columns_mean(X), N)
        Sigma = diag(R.T * R).T / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Z = hadamard(row_repeat(inv_sqrt_Sigma, N), R)
        Y = hadamard(row_repeat(gamma, N), Z) + row_repeat(beta, N)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(row_repeat(gamma, N), DY)
        Dbeta = columns_sum(DY)
        Dgamma = columns_sum(hadamard(DY, Z))
        DX = hadamard(row_repeat(inv_sqrt_Sigma / N, N), (N * identity(N) - ones(N, N)) * DZ - hadamard(Z, row_repeat(diag(Z.T * DZ).T, N)))

        # test gradients
        DX1 = gradient(loss(Y), x)
        Dbeta1 = gradient(loss(Y), beta)
        Dgamma1 = gradient(loss(Y), gamma)
        Y_z = hadamard(row_repeat(gamma, N), z) + row_repeat(beta, N)
        DZ1 = substitute(gradient(loss(Y_z), z), (z, Z))

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))
        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_batch_normalization_layer(self):
        self._test_batch_normalization_layer(2, 3, loss=elements_sum)
        self._test_batch_normalization_layer(3, 2, loss=elements_sum)

    def _test_yeh_batch_normalization_layer(self, D, N, loss):
        # see https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html
        K = D  # K and D are always equal in batch normalization

        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        beta = matrix('beta', 1, K)
        gamma = matrix('gamma', 1, K)
        X = x

        # feedforward
        R = X - row_repeat(columns_mean(X), N)
        Sigma = diag(R.T * R).T / N
        inv_sqrt_Sigma = inv_sqrt(Sigma)
        Z = hadamard(row_repeat(inv_sqrt_Sigma, N), R)
        Y = hadamard(row_repeat(gamma, N), Z) + row_repeat(beta, N)

        # symbolic differentiation
        DY = substitute(gradient(loss(y), y), (y, Y))

        # backpropagation
        DZ = hadamard(row_repeat(gamma, N), DY)  # this equation is not explicitly given in [Yeh 2017]
        Dbeta = columns_sum(DY)                  # this equation is the same as in [Yeh 2017]
        Dgamma = columns_sum(hadamard(DY, Z))    # I can't parse the equation in [Yeh 2017], but this is probably it
        DX = (1 / N) * (-hadamard(row_repeat(Dgamma, N), Z) + N * DY - row_repeat(Dbeta, N)) * row_repeat(hadamard(gamma, Sigma), D) # I can't parse the equation in [Yeh 2017], but this is probably it

        # test gradients
        DX1 = gradient(loss(Y), x)
        Dbeta1 = gradient(loss(Y), beta)
        Dgamma1 = gradient(loss(Y), gamma)
        Y_z = hadamard(row_repeat(gamma, N), z) + row_repeat(beta, N)
        DZ1 = substitute(gradient(loss(Y_z), z), (z, Z))

        self.assertTrue(equal_matrices(DX, DX1))
        self.assertTrue(equal_matrices(Dbeta, Dbeta1))
        self.assertTrue(equal_matrices(Dgamma, Dgamma1))
        self.assertTrue(equal_matrices(DZ, DZ1))

    def test_yeh_batch_normalization_layer(self):
        self._test_yeh_batch_normalization_layer(2, 3, loss=elements_sum)
        self._test_yeh_batch_normalization_layer(3, 2, loss=elements_sum)


if __name__ == '__main__':
    import unittest
    unittest.main()