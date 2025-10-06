#!/usr/bin/env python3
# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from nerva_sympy.loss_functions import Softmax_cross_entropy_loss, Cross_entropy_loss

from nerva_sympy.matrix_operations import *
from nerva_sympy.softmax_functions import *

from utilities import equal_matrices, matrix, to_matrix, to_number

Matrix = sp.Matrix


class TestSoftmaxDerivation(TestCase):
    # section 4
    def test_dsoftmax_dz_derivation(self):
        K = 3

        z = matrix('z', 1, K)
        y = softmax(z)

        dsoftmax_dz = softmax(z).jacobian(z)
        lhs = exp(z)
        rhs = reciprocal(rows_sum(exp(z)))
        dlhs_dz = lhs.jacobian(z)
        drhs_dz = rhs.jacobian(z)
        self.assertTrue(equal_matrices(dsoftmax_dz, dlhs_dz * to_number(rhs) + lhs.T * drhs_dz))
        self.assertTrue(equal_matrices(dlhs_dz * to_number(rhs), Diag(exp(z)) * to_number(rhs)))
        R = to_number(rows_sum(exp(z)))
        self.assertTrue(equal_matrices(drhs_dz, - exp(z) / (R * R)))
        self.assertTrue(equal_matrices(Diag(exp(z)) * to_number(rhs), Diag(y)))
        self.assertTrue(equal_matrices(exp(z).T * (exp(z) / (R * R)), y.T * y))
        self.assertTrue(equal_matrices(dsoftmax_dz, Diag(y) - y.T * y))

    def test_softmax_definition_section_5_1(self):
        K = 3
        N = 4

        z = matrix('z', 1, K)
        self.assertTrue(softmax(z).equals(exp(z) / to_number(exp(z) * ones(K))))
        for i in range(K):
            self.assertEqual(softmax(z)[i], sp.exp(z[i]) / sum(sp.exp(z[k]) for k in range(K)))

        Z = matrix('Z', N, K)
        self.assertTrue(softmax(Z).equals(hadamard(exp(Z), reciprocal(exp(Z) * ones(K)) * ones(K).T)))
        for i in range(N):
            self.assertTrue(softmax(Z).row(i).equals(softmax(Z.row(i))))

    def test_softmax_derivation_section_5_1(self):
        K = 3

        z = matrix('z', 1, K)
        y = softmax(z)

        e1 = softmax(z).jacobian(z)
        e2 = (exp(z) / to_number(exp(z) * ones(K))).jacobian(z)
        e3 = exp(z).jacobian(z) / to_number(exp(z) * ones(K)) + exp(z).T * reciprocal(exp(z) * ones(K)).jacobian(z)
        e4 = Diag(exp(z)) / to_number(exp(z) * ones(K)) - exp(z).T * exp(z) / (to_number(exp(z) * ones(K))**2)
        e5 = Diag(exp(z) / to_number(exp(z) * ones(K))) - (exp(z) / to_number(exp(z) * ones(K))).T * exp(z) / to_number(exp(z) * ones(K))
        e6 = Diag(y) - y.T * y

        self.assertTrue(e1.equals(e2))
        self.assertTrue(e2.equals(e3))
        self.assertTrue(e3.equals(e4))
        self.assertTrue(e4.equals(e5))
        self.assertTrue(e5.equals(e6))

    def test_softmax_gradient_section_5_1(self):
        K = 3
        N = 4
        Z = matrix('Z', N, K)
        Y = matrix('Y', N, K)
        T = matrix('T', N, K)
        L = Cross_entropy_loss

        for i in range(N):
            dL_dz_i = gradient(L(softmax(Z), T), Z.row(i))
            dL_dy_i = substitute(gradient(L(Y, T), Y.row(i)), [(Y, softmax(Z))])
            dsoftmax_dz_i = softmax(Z.row(i)).jacobian(Z.row(i))
            self.assertTrue(dL_dz_i.equals(dL_dy_i * dsoftmax_dz_i))

            y_i = softmax(Z.row(i))
            Dy_i = dL_dy_i
            Dz_i = dL_dz_i
            e1 = Dz_i
            e2 = Dy_i * (Diag(y_i) - y_i.T * y_i)
            e3 = hadamard(Dy_i, y_i) - Dy_i * y_i.T * y_i
            self.assertTrue(e1.equals(e2))
            self.assertTrue(e2.equals(e3))

        DY = substitute(gradient(L(Y, T), Y), [(Y, softmax(Z))])
        DZ = substitute(hadamard(DY - diag(DY * Y.T) * ones(K).T, Y), [(Y, softmax(Z))])
        DZ1 = gradient(L(softmax(Z), T), Z)
        self.assertTrue(DZ.equals(DZ1))

    def test_softmax_derivation_appendix_C1(self):
        D = 3
        N = 4

        self.assertEqual(ones(D).shape, (D, 1))

        x = matrix('x', 1, D)
        z = x - sp.Max(*x) * ones(D).T
        self.assertEqual(softmax(x), exp(x) / to_number(exp(x) * ones(D)))
        self.assertEqual(stable_softmax(x), exp(z) / to_number(exp(z) * ones(D)))
        self.assertEqual(log_softmax(x), x - log((exp(x) * ones(D))) * ones(D).T)
        self.assertEqual(stable_log_softmax(x), z - log((exp(z)* ones(D))) * ones(D).T)

        X = matrix('x', N, D)
        Z = X - rows_max(X) * ones(D).T
        self.assertEqual(softmax(X), hadamard(exp(X), (reciprocal(exp(X) * ones(D)) * ones(D).T)))
        self.assertEqual(stable_softmax(X), hadamard(exp(Z), reciprocal(exp(Z) * ones(D)) * ones(D).T))
        self.assertEqual(log_softmax(X), X - log((exp(X) * ones(D))) * ones(D).T)
        self.assertEqual(stable_log_softmax(X), Z - log((exp(Z)* ones(D))) * ones(D).T)

        # derivative of softmax(x)
        e1 = softmax(x).jacobian(x)
        e2 = stable_softmax(x).jacobian(x)
        e3 = Diag(softmax(x)) - softmax(x).T * softmax(x)
        self.assertTrue(equal_matrices(e1, e2))
        self.assertTrue(equal_matrices(e2, e3))

        # derivative of log_softmax(x)
        f1 = log_softmax(x).jacobian(x)
        f2 = stable_log_softmax(x).jacobian(x)
        f3 = identity(D) - ones(D) * softmax(x)
        self.assertTrue(equal_matrices(f1, f2))
        self.assertTrue(equal_matrices(f2, f3))

    def test_softmax_derivation_appendix_C2(self):
        D = 3

        x = matrix('x', 1, D)
        y = matrix('y', 1, D)

        e1 = log(softmax(x)).jacobian(x)
        e2 = log(y).jacobian(y) * softmax(x).jacobian(x)
        e3 = Diag(reciprocal(y)) * (Diag(y) - y.T * y)
        e4 = identity(D) - Diag(reciprocal(y)) * y.T * y
        e5 = identity(D) - ones(D) * y
        e6 = identity(D) - ones(D) * softmax(x)

        # replace occurrences of y by softmax(x)
        e2 = substitute(e2, (y, softmax(x)))
        e3 = substitute(e3, (y, softmax(x)))
        e4 = substitute(e4, (y, softmax(x)))
        e5 = substitute(e5, (y, softmax(x)))

        self.assertTrue(equal_matrices(e1, e2))
        self.assertTrue(equal_matrices(e2, e3))
        self.assertTrue(equal_matrices(e3, e4))
        self.assertTrue(equal_matrices(e4, e5))
        self.assertTrue(equal_matrices(e5, e6))

    def test1(self):
        K = 3
        z = matrix('z', 1, K)
        y = softmax(z)
        self.assertTrue(equal_matrices(softmax(z).jacobian(z), Diag(y) - y.T * y))

    def test2(self):
        K = 3
        y = matrix('y', 1, K)
        z = matrix('z', 1, K)
        L1 = lambda Y: to_matrix(elements_sum(Y))
        L2 = lambda Y: to_matrix(elements_sum(hadamard(Y, Y)))

        for L in [L1, L2]:
            y_z = softmax(z)
            dsoftmax_z_dz = softmax(z).jacobian(z)
            dL_dy = substitute(L(y).jacobian(y), (y, y_z))
            dL_dz = L(y_z).jacobian(z)
            self.assertTrue(equal_matrices(dL_dz, dL_dy * dsoftmax_z_dz))

    def test3(self):
        K = 3
        y = matrix('y', 1, K)
        z = matrix('z', 1, K)
        L1 = lambda Y: to_matrix(elements_sum(Y))
        L2 = lambda Y: to_matrix(elements_sum(hadamard(Y, Y)))

        for L in [L1, L2]:
            y_z = softmax(z)
            dsoftmax_z_dz = y_z.jacobian(z)
            dL_dy = substitute(L(y).jacobian(y), (y, y_z))
            dL_dz = L(y_z).jacobian(z)
            Dy = dL_dy
            Dz = dL_dz
            self.assertTrue(equal_matrices(Dz, Dy * dsoftmax_z_dz))
            self.assertTrue(equal_matrices(dsoftmax_z_dz, Diag(y_z) - y_z.T * y_z))
            self.assertTrue(equal_matrices(Dz, Dy * (Diag(y_z) - y_z.T * y_z)))
            self.assertTrue(equal_matrices(Dz, hadamard(y_z, Dy) - Dy * y_z.T * y_z))

    def test4(self):
        D = 2
        x = matrix('x', 1, D)
        y = rows_sum(exp(x))
        dy_dx = jacobian(y, x)
        dy_dx_expected = exp(x)
        self.assertEqual(dy_dx_expected, dy_dx)

    def test5(self):
        D = 2
        x = matrix('x', 1, D)
        y = log(rows_sum(exp(x)))
        dy_dx = jacobian(y, x)
        dy_dx_expected = reciprocal(rows_sum(exp(x))) * exp(x)
        self.assertEqual(dy_dx_expected, dy_dx)

    def test6(self):
        D = 2
        x = matrix('x', 1, D)
        y = log(reciprocal(rows_sum(exp(x))) * exp(x))
        y1 = x - log(rows_sum(exp(x)))  * ones(1, D)
        y = sp.simplify(y)
        y1 = sp.simplify(y1)
        self.assertEqual(sp.simplify(y), sp.simplify(y1))

    def test7(self):
        D = 2
        x = matrix('x', 1, D)
        y = log(reciprocal(rows_sum(exp(x))) * exp(x))
        y = sp.simplify(y)
        dy_dx = jacobian(y, x)
        dy_dx_expected = identity(D) - row_repeat(reciprocal(rows_sum(exp(x))) * exp(x), D)
        dy_dx = sp.simplify(dy_dx)
        dy_dx_expected = sp.simplify(dy_dx_expected)
        self.assertEqual(dy_dx_expected, dy_dx)

    def test_example(self):
        from sympy import symarray
        K = 3
        z = Matrix(symarray('z', (1, K), real=True))  # create a symbolic 1xK vector
        y = softmax(z)
        e = exp(z)
        f = rows_sum(e)[0, 0]  # rows_sum returns a Matrix, so we extract the value
        self.assertTrue(equal_matrices(softmax(z).jacobian(z), Diag(e) / f - (e.T * e) / (f * f)))
        self.assertTrue(equal_matrices(softmax(z).jacobian(z), Diag(y) - y.T * y))


if __name__ == '__main__':
    import unittest
    unittest.main()