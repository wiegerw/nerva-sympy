#!/usr/bin/env python3

# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# see also https://docs.sympy.org/latest/modules/matrices/matrices.html

from unittest import TestCase

from nerva_sympy.matrix_operations import *

from utilities import matrix

Matrix = sp.Matrix


class TestBatchNormalizationDefinition(TestCase):
    def test_batch_norm_definition_section_3_1(self):
        N = 4
        D = 3
        X = matrix('x', N, D)
        beta = matrix('beta', 1, D)
        gamma = matrix('gamma', 1, D)

        R = X - (ones(N, N) / N) * X
        Sigma = diag(R.T * R).T / N
        mu = (ones(N).T * X) / N
        Z = hadamard(ones(N) * inv_sqrt(Sigma), R)

        self.assertTrue(R.shape == (N, D))
        self.assertTrue(Sigma.shape == (1, D))
        self.assertTrue(mu.shape == (1, D))

        r = lambda i: X.row(i) - mu                        # r(i) is the i-th row of R
        for i in range(N):
            self.assertEqual(r(i), R.row(i))

        sigma2 = lambda j: (R.col(j).T * R.col(j) / N)[0]  # sigma2(j) is the j-th element of Sigma
        for j in range(D):
            self.assertEqual(sigma2(j), Sigma[0, j])

        Y = hadamard(ones(N) * gamma, Z) + ones(N) * beta
        for i in range(N):
            z_i = Z.row(i)
            y_i = hadamard(gamma, z_i) + beta
            self.assertEqual(y_i, Y.row(i))


if __name__ == '__main__':
    import unittest
    unittest.main()