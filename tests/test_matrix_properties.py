#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from nerva_sympy.matrix_operations import *
from utilities import matrix, to_number


# Appendix A
class TestMatrixProperties(TestCase):
    def test_property1(self):
        m = 3
        n = 4
        X = matrix('X', m, n)
        Y = matrix('Y', m, n)
        Z1 = join_columns([X.col(j) * X.col(j).T * Y.col(j) for j in range(n)])
        Z2 = hadamard(X, ones(m) * diag(X.T * Y).T)
        self.assertTrue(Z1.equals(Z2))

    def test_property2(self):
        m = 3
        n = 4
        X = matrix('X', m, n)
        Y = matrix('Y', m, n)
        Z1 = join_columns([ones(m) * X.col(j).T * Y.col(j) for j in range(n)])
        Z2 = ones(m) * diag(X.T * Y).T
        self.assertTrue(Z1.equals(Z2))

    def test_property3(self):
        m = 3
        n = 4
        X = matrix('X', m, n)
        Y = matrix('Y', m, n)
        Z1 = join_columns([X.col(j) * ones(m).T * Y.col(j) for j in range(n)])
        Z2 = hadamard(X, (ones(m) * ones(m).T * Y))
        self.assertTrue(Z1.equals(Z2))

    def test_property4(self):
        m = 3
        n = 4
        X = matrix('X', m, n)
        Y = matrix('Y', m, n)
        Z1 = join_rows([X.row(i) * Y.row(i).T * Y.row(i) for i in range(m)])
        Z2 = hadamard(diag(X * Y.T) * ones(n).T, Y)
        self.assertTrue(Z1.equals(Z2))

    def test_property5(self):
        m = 3
        n = 4
        X = matrix('X', m, n)
        Y = matrix('Y', m, n)
        Z1 = join_rows([X.row(i) * Y.row(i).T * ones(n).T for i in range(m)])
        Z2 = diag(X * Y.T) * ones(n).T
        self.assertTrue(Z1.equals(Z2))

    def test_property6(self):
        m = 3
        n = 4
        X = matrix('X', m, n)
        Y = matrix('Y', m, n)
        Z1 = join_rows([X.row(i) * ones(n) * Y.row(i) for i in range(m)])
        Z2 = hadamard(X * ones(n) * ones(n).T, Y)
        self.assertTrue(Z1.equals(Z2))

    def test_property4_derivation(self):
        m = 3
        n = 4
        X = matrix('X', m, n)
        Y = matrix('Y', m, n)
        Z = join_rows([X.row(i) * Y.row(i).T * Y.row(i) for i in range(m)])
        for i in range(m):
            x_i = X.row(i)
            y_i = Y.row(i)
            z_i = Z.row(i)
            for j in range(n):
                y_ij = y_i[j]
                z_ij = z_i[j]
                self.assertEqual(z_ij, to_number(x_i * y_i.T) * y_ij)
        R = Matrix([[to_number(X.row(i) * Y.row(i).T) for j in range(n)] for i in range(m)])
        self.assertTrue(Z.equals(hadamard(R, Y)))
        rhs = Matrix([[to_number(X.row(i) * Y.row(i).T)] for i in range(m)])
        self.assertTrue(diag(X * Y.T).equals(rhs))
        self.assertTrue(R.equals(diag(X * Y.T) * ones(n).T))


class TestDerivatives(TestCase):
    def test_derivative_gx_x(self):
        n = 3
        x = Matrix(sp.symbols('x:{}'.format(n))).T
        self.assertTrue(is_row_vector(x))

        g = sp.Function('g', real=True)(*x)
        J1 = jacobian(g * x, x)
        J2 = x.T * jacobian(Matrix([[g]]), x) + g * sp.eye(n)
        self.assertTrue(J1.equals(J2))


if __name__ == '__main__':
    import unittest
    unittest.main()