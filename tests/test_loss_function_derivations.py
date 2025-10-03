#!/usr/bin/env python3
# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from nerva_sympy.activation_functions import sigmoid

from nerva_sympy.loss_functions import cross_entropy_loss, softmax_cross_entropy_loss, logistic_cross_entropy_loss, \
    squared_error_loss, Squared_error_loss, Cross_entropy_loss, Softmax_cross_entropy_loss, Logistic_cross_entropy_loss, \
    negative_log_likelihood_loss, Negative_log_likelihood_loss
from nerva_sympy.matrix_operations import *
from nerva_sympy.softmax_functions import *

from utilities import equal_matrices, matrix, to_matrix, to_number

Matrix = sp.Matrix


# appendix D
class TestLossFunctionDerivation(TestCase):
    def test_squared_error_loss(self):
        K = 3
        N = 4

        y = matrix('y', 1, K)
        t = matrix('t', 1, K)

        self.assertEqual(squared_error_loss(y, t), to_number((y - t) * (y - t).T))
        self.assertEqual(gradient(squared_error_loss(y, t), y), 2 * (y - t))

        Y = matrix('Y', N, K)
        T = matrix('T', N, K)
        self.assertEqual(Squared_error_loss(Y, T), to_number(ones(N).T * (hadamard((Y - T), (Y - T))) * ones(K)))

        DY = 2 * (Y - T)
        self.assertEqual(DY.shape, (N, K))
        for i in range(N):
            y = Y.row(i)
            t = T.row(i)
            e1 = gradient(squared_error_loss(y, t), y)
            e2 = DY.row(i)
            self.assertTrue(equal_matrices(e1, e2))

    def test_cross_entropy_loss(self):
        K = 3
        N = 4

        y = matrix('y', 1, K)
        t = matrix('t', 1, K)

        self.assertEqual(cross_entropy_loss(y, t), to_number(-t * log(y).T))
        self.assertEqual(gradient(cross_entropy_loss(y, t), y), -hadamard(t, reciprocal(y)))

        Y = matrix('Y', N, K)
        T = matrix('T', N, K)
        self.assertEqual(Cross_entropy_loss(Y, T), to_number(-ones(N).T * (hadamard(T, log(Y))) * ones(K)))

        DY = -hadamard(T, reciprocal(Y))
        self.assertEqual(DY.shape, (N, K))
        for i in range(N):
            y = Y.row(i)
            t = T.row(i)
            e1 = gradient(cross_entropy_loss(y, t), y)
            e2 = DY.row(i)
            self.assertTrue(equal_matrices(e1, e2))

    def test_softmax_cross_entropy_loss(self):
        K = 3
        N = 4

        y = matrix('y', 1, K)
        t = matrix('t', 1, K)

        self.assertEqual(softmax_cross_entropy_loss(y, t), to_number(-t * log_softmax(y).T))
        self.assertTrue(equal_matrices(gradient(softmax_cross_entropy_loss(y, t), y), t * ones(K) * softmax(y) - t))

        Y = matrix('Y', N, K)
        T = matrix('T', N, K)
        self.assertEqual(Softmax_cross_entropy_loss(Y, T), to_number(-ones(N).T * (hadamard(T, log_softmax(Y))) * ones(K)))

        DY = hadamard(softmax(Y), T * ones(K) * ones(K).T) - T
        self.assertEqual(DY.shape, (N, K))
        for i in range(N):
            y = Y.row(i)
            t = T.row(i)
            e1 = gradient(softmax_cross_entropy_loss(y, t), y)
            e2 = DY.row(i)
            self.assertTrue(equal_matrices(e1, e2))

    def test_logistic_cross_entropy_loss(self):
        K = 3
        N = 4

        y = matrix('y', 1, K)
        t = matrix('t', 1, K)

        self.assertEqual(logistic_cross_entropy_loss(y, t), to_number(-t * log(Sigmoid(y)).T))
        self.assertTrue(equal_matrices(gradient(logistic_cross_entropy_loss(y, t), y), hadamard(t, Sigmoid(y)) - t))

        Y = matrix('Y', N, K)
        T = matrix('T', N, K)
        self.assertEqual(Logistic_cross_entropy_loss(Y, T), to_number(-ones(N).T * (hadamard(T, log(Sigmoid(Y))) * ones(K))))

        DY = hadamard(T, Sigmoid(Y)) - T
        self.assertEqual(DY.shape, (N, K))
        for i in range(N):
            y = Y.row(i)
            t = T.row(i)
            e1 = gradient(logistic_cross_entropy_loss(y, t), y)
            e2 = DY.row(i)
            self.assertTrue(equal_matrices(e1, e2))

    def test_negative_log_likelihood_loss(self):
        K = 3
        N = 4

        y = matrix('y', 1, K)
        t = matrix('t', 1, K)

        self.assertEqual(negative_log_likelihood_loss(y, t), to_number(-log(y * t.T)))
        self.assertEqual(gradient(Negative_log_likelihood_loss(y, t), y), -reciprocal(y * t.T) * t)

        Y = matrix('Y', N, K)
        T = matrix('T', N, K)
        self.assertEqual(Negative_log_likelihood_loss(Y, T), to_number(-ones(N).T * log(hadamard(Y, T) * ones(K))))

        DY = -hadamard(reciprocal(hadamard(Y, T) * ones(K)) * ones(K).T, T)
        self.assertEqual(DY.shape, (N, K))
        for i in range(N):
            y = Y.row(i)
            t = T.row(i)
            e1 = gradient(negative_log_likelihood_loss(y, t), y)
            e2 = DY.row(i)
            self.assertTrue(equal_matrices(e1, e2))


if __name__ == '__main__':
    import unittest
    unittest.main()