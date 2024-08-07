#!/usr/bin/env python3
# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from nerva_sympy.loss_functions import cross_entropy_loss, softmax_cross_entropy_loss, logistic_cross_entropy_loss
from nerva_sympy.matrix_operations import *
from nerva_sympy.softmax_functions import *

from utilities import equal_matrices, matrix, to_matrix

Matrix = sp.Matrix


class TestLossFunctionDerivation(TestCase):
    # appendix D1
    def test_cross_entropy_loss(self):
        K = 3

        y = matrix('y', 1, K)
        t = matrix('t', 1, K)

        L = to_matrix(cross_entropy_loss(y, t))

        dL_dy = L.jacobian(y)
        self.assertTrue(equal_matrices(dL_dy, -t * Diag(reciprocal(y))))
        self.assertTrue(equal_matrices(-t * Diag(reciprocal(y)), -hadamard(t, reciprocal(y))))

    # appendix D1
    def test_softmax_cross_entropy_loss(self):
        K = 3

        y = matrix('y', 1, K)
        t = matrix('t', 1, K)

        L = to_matrix(softmax_cross_entropy_loss(y, t))

        dL_dy = L.jacobian(y)
        dlog_softmax_dy = log_softmax(y).jacobian(y)
        Z = identity(K) - row_repeat(softmax(y), K)
        self.assertTrue(equal_matrices(dL_dy, -t * dlog_softmax_dy))
        self.assertTrue(equal_matrices(dlog_softmax_dy, Z))
        self.assertTrue(equal_matrices(-t * dlog_softmax_dy, -t * Z))
        self.assertTrue(equal_matrices(-t * Z, rows_sum(t) * softmax(y) - t))

    # appendix D1
    def test_logistic_cross_entropy_loss(self):
        K = 3

        y = matrix('y', 1, K)
        t = matrix('t', 1, K)

        L = to_matrix(logistic_cross_entropy_loss(y, t))

        dL_dy = L.jacobian(y)
        dlog_sigmoid_dy = log_sigmoid(y).jacobian(y)
        self.assertTrue(equal_matrices(dL_dy, -t * dlog_sigmoid_dy))
        self.assertTrue(equal_matrices(-t * dlog_sigmoid_dy, -t * Diag(ones(K).T - Sigmoid(y))))
        self.assertTrue(equal_matrices(-t * Diag(ones(K).T - Sigmoid(y)), hadamard(t, Sigmoid(y)) - t))


if __name__ == '__main__':
    import unittest
    unittest.main()