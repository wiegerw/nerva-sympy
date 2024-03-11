#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from nerva_sympy.matrix_operations import *


class TestLemmas(TestCase):
    def test_lemma1(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_rows([X.row(i) * Y.row(i).T * Y.row(i) for i in range(m)])
        Z2 = hadamard(Y, column_repeat(diag(X * Y.T), n))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma2(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_rows([X.row(i) * column_repeat(Y.row(i).T, n) for i in range(m)])
        Z2 = column_repeat(diag(X * Y.T), n)
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma3(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = join_rows([X.row(i) * row_repeat(Y.row(i), n) for i in range(m)])
        Z2 = hadamard(Y, column_repeat(rows_sum(X), n))
        self.assertEqual(sp.simplify(Z1 - Z2), sp.zeros(m, n))

    def test_lemma4(self):
        m = 2
        n = 3

        X = Matrix(sp.symarray('x', (m, n), real=True))
        Y = Matrix(sp.symarray('y', (m, n), real=True))
        Z1 = sum(dot(X.row(i).T, Y.row(i).T) for i in range(m))
        Z2 = elements_sum(hadamard(X, Y))
        self.assertEqual(Z1, Z2)


class TestDerivatives(TestCase):
    def test_derivative_gx_x(self):
        n = 3
        x = Matrix(sp.symbols('x:{}'.format(n))).T
        self.assertTrue(is_row_vector(x))

        g = sp.Function('g', real=True)(*x)
        J1 = jacobian(g * x, x)
        J2 = x.T * jacobian(Matrix([[g]]), x) + g * sp.eye(n)
        self.assertEqual(sp.simplify(J1 - J2), sp.zeros(n, n))


if __name__ == '__main__':
    import unittest
    unittest.main()