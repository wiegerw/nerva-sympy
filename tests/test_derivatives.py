#!/usr/bin/env python3
# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from nerva_sympy.matrix_operations import *

from utilities import equal_matrices, matrix, pp, to_matrix

Matrix = sp.Matrix


class TestLemmas(TestCase):
    def test_chain_rule(self):
        x = matrix('x', 1, 3)
        y = matrix('y', 1, 3)
        c = Matrix([[2, 3, 4]])
        f_x = x * Diag(c) + Matrix([[2 * x[0, 1], x[0, 1], x[0, 0]]])
        g_y = y * y.T
        g_fx = substitute(g_y, (y, f_x))

        pp('x', x)
        pp('f(x)', f_x)
        pp('g(f(x))', g_fx)
        pp('g(y)', g_y)
        pp('f(x).jacobian(x)', f_x.jacobian(x))
        pp('g(f(x)).jacobian(x)', g_fx.jacobian(x))
        pp('g(y).jacobian(y)', g_y.jacobian(y))

        df_dx = f_x.jacobian(x)
        dg_dy = substitute(g_y.jacobian(y), (y, f_x))
        dg_dx = g_fx.jacobian(x)

        pp('dg_dx', dg_dx)
        pp('dg_dy * df_dx', dg_dy * df_dx)
        self.assertEqual(dg_dx, dg_dy * df_dx)

    def test_lemma_fx_x(self):
        x = matrix('x', 1, 3)
        f = lambda x: 3 * x[0, 0] + 2 * x[0, 1] + 5 * x[0, 2]
        g = lambda x: f(x) * x
        f_x = f(x)
        g_x = g(x)

        pp('x', x)
        print('f(x)', f_x)
        pp('g(x)', g_x)

        df_dx = sp.Matrix([[f_x]]).jacobian(x)
        dg_dx = g_x.jacobian(x)

        pp('df_dx', df_dx)
        pp('dg_dx', dg_dx)
        self.assertEqual(dg_dx, x.T * df_dx + f_x * sp.eye(3))


class TestProductRule(TestCase):
    def test_product_rule1(self):
        x1, x2, x3 = sp.symbols(['x1', 'x2', 'x3'], Real=True)
        x = Matrix([x1, x2, x3])
        f = Matrix([2 * x1 * x2 + 3 * x1 - x2, x1 * x1 - 3 * x1 * x2 + 4 * x3])
        g_ = x1 + 5 * x2 + x2 * x3
        g = Matrix([g_])
        h = f * g_
        dh_dx = h.jacobian(x)
        df_dx = f.jacobian(x)
        dg_dx = to_matrix(g).jacobian(x)
        rhs = df_dx * g_ + f * dg_dx
        self.assertTrue(equal_matrices(dh_dx, rhs))

    def test_product_rule2(self):
        x1, x2, x3 = sp.symbols(['x1', 'x2', 'x3'], Real=True)
        x = Matrix([x1, x2, x3])
        f = Matrix([[2 * x1 * x2 + 3 * x1 - x2, x1 * x1 - 3 * x1 * x2 + 4 * x3]])
        g_ = x1 + 5 * x2 + x2 * x3
        g = Matrix([g_])
        h = f * g_
        dh_dx = h.jacobian(x)
        df_dx = f.jacobian(x)
        dg_dx = to_matrix(g).jacobian(x)
        rhs = df_dx * g_ + f.T * dg_dx
        self.assertTrue(equal_matrices(dh_dx, rhs))

    def test_product_rule3(self):
        m, n = 2, 3
        x1, x2, x3 = sp.symbols(['x1', 'x2', 'x3'], Real=True)
        x = Matrix([x1, x2, x3])
        A = matrix('A', m, n)
        f = Matrix([[2 * x1 * x2 + 3 * x1 - x2, x1 * x1 - 3 * x1 * x2 + 4 * x3]])
        g = f * A
        df_dx = f.jacobian(x)
        dg_dx = g.jacobian(x)
        rhs = A.T * df_dx
        self.assertTrue(equal_matrices(dg_dx, rhs))

    def test_product_rule4(self):
        m, n = 2, 3
        x1, x2, x3 = sp.symbols(['x1', 'x2', 'x3'], Real=True)
        x = Matrix([x1, x2, x3])
        A = matrix('A', m, n)
        f = Matrix([2 * x1 * x2 + 3 * x1 - x2, x1 * x1 - 3 * x1 * x2 + 4 * x3, 2 * x1 + 3 * x2 + 4 * x3])
        g = A * f
        df_dx = f.jacobian(x)
        dg_dx = g.jacobian(x)
        rhs = A * df_dx
        self.assertTrue(equal_matrices(dg_dx, rhs))


if __name__ == '__main__':
    import unittest
    unittest.main()