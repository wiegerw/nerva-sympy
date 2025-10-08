#!/usr/bin/env python3

# Copyright 2024 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

# see also https://docs.sympy.org/latest/modules/matrices/matrices.html

from unittest import TestCase

from nerva_sympy.loss_functions import Squared_error_loss, Cross_entropy_loss
from nerva_sympy.matrix_operations import *
from nerva_sympy.softmax_functions import softmax, log_softmax

from utilities import equal_matrices, matrix, to_matrix, to_number, sum1

Matrix = sp.Matrix


def squared_error_rows(X: Matrix):
    m, n = X.shape

    def f(x: Matrix) -> float:
        return sp.sqrt(sum(xj * xj for xj in x))

    return sum(f(X.row(i)) for i in range(m))


def squared_error_columns(X: Matrix):
    m, n = X.shape

    def f(x: Matrix) -> float:
        return sp.sqrt(sum(xj * xj for xj in x))

    return sum(f(X.col(j)) for j in range(n))


class TestLinearLayerDerivation(TestCase):
    def test_derivations(self):
        N = 3
        D = 4
        K = 2

        x = matrix('x', N, D)
        y = matrix('y', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        X = x
        W = w

        # feedforward
        Y = X * W.T + row_repeat(b, N)

        L = lambda Y: to_matrix(squared_error_columns(Y))

        i = 1
        x_i = x.row(i)  # 1 x D
        y_i = y.row(i)  # 1 x K

        DY = substitute(gradient(L(y), y), (y, Y))
        DW = DY.T * X

        L_x = L(Y)
        L_y = L(y)

        dL_dx_i = L_x.jacobian(x_i)
        dL_dy_i = substitute(L_y.jacobian(y_i), [(y, Y)])
        dy_i_dx_i = Y.row(i).jacobian(x_i)

        # first derivation
        self.assertTrue(equal_matrices(dL_dx_i, dL_dy_i * dy_i_dx_i))
        self.assertTrue(equal_matrices(dL_dy_i * dy_i_dx_i, dL_dy_i * W))

        # second derivation
        dL_db = L_x.jacobian(b)
        sum_dL_dyi = substitute(sum([L_y.jacobian(y.row(i)) * Y.row(i).jacobian(b) for i in range(N)], sp.zeros(1, K)), [(y, Y)])
        self.assertTrue(equal_matrices(dL_db, sum_dL_dyi))
        for i in range(N):
            self.assertTrue(equal_matrices(Y.row(i).jacobian(b), sp.eye(K)))
        self.assertTrue(equal_matrices(sum_dL_dyi, columns_sum(DY)))

        # third derivation
        i = 1
        j = 2
        k = 1

        e_i = identity(K).col(i)  # unit column vector with a 1 on the i-th position
        y_i = Y.row(i)  # 1 x K
        w_i = w.row(i)
        dyi_dwi = y_i.jacobian(w_i)
        self.assertTrue(equal_matrices(dyi_dwi, e_i * x_i))

        x_k = x.row(k)
        y_k = Y.row(k)
        self.assertTrue(equal_matrices(y_k, x_k * w.T + b))

        w_ij = to_matrix(w[i, j])
        dyk_dwij = y_k.jacobian(w_ij)
        self.assertTrue(equal_matrices(dyk_dwij, x[k, j] * e_i))

    def test_derivation_DX(self):
        N = 3
        D = 4
        K = 2

        X = matrix('X', N, D)
        W = matrix('D', K, D)
        Y = matrix('Y', N, K)
        b = matrix('b', 1, K)
        T = matrix('T', N, K)

        Y_x = X * W.T + ones(N) * b
        L_x = Squared_error_loss(Y_x, T)
        L_y = Squared_error_loss(Y, T)

        # replace Y by Y_x
        def expand(x):
            return substitute(x, [(Y, Y_x)])

        for i in range(N):
            Dx_i = gradient(L_x, X.row(i))
            Dy_i = expand(gradient(L_y, Y.row(i)))
            e1 = Dx_i
            e2 = sum((expand(gradient(L_y, Y.row(n))) * Y_x.row(n).jacobian(X.row(i)) for n in range(N)), sp.zeros(1, D))
            e3 = Dy_i * Y_x.row(i).jacobian(X.row(i))
            e4 = Dy_i * W
            self.assertTrue(e1.equals(e2))
            self.assertTrue(e2.equals(e3))
            self.assertTrue(e3.equals(e4))

    def test_derivation_Db(self):
        N = 3
        D = 4
        K = 2

        X = matrix('X', N, D)
        W = matrix('D', K, D)
        Y = matrix('Y', N, K)
        b = matrix('b', 1, K)
        T = matrix('T', N, K)

        Y_x = X * W.T + ones(N) * b
        L_x = Squared_error_loss(Y_x, T)
        L_y = Squared_error_loss(Y, T)

        # replace Y by Y_x
        def expand(x):
            return substitute(x, [(Y, Y_x)])

        DY = join_rows([expand(gradient(L_y, Y.row(n))) for n in range(N)])

        for i in range(N):
            Db = gradient(L_x, b)
            e1 = Db
            e2 = sum((expand(gradient(L_y, Y.row(n))) * Y_x.row(n).jacobian(b) for n in range(N)), sp.zeros(1, K))
            e3 = sum((expand(gradient(L_y, Y.row(n))) * identity(K) for n in range(N)), sp.zeros(1, K))
            e4 = sum((expand(gradient(L_y, Y.row(n))) for n in range(N)), sp.zeros(1, K))
            e5 = ones(N).T * DY
            self.assertTrue(e1.equals(e2))
            self.assertTrue(e2.equals(e3))
            self.assertTrue(e3.equals(e4))
            self.assertTrue(e4.equals(e5))

    def test_derivation_DW(self):
        N = 3
        D = 4
        K = 2

        X = matrix('X', N, D)
        W = matrix('D', K, D)
        Y = matrix('Y', N, K)
        b = matrix('b', 1, K)
        T = matrix('T', N, K)

        Y_x = X * W.T + ones(N) * b
        L_x = Squared_error_loss(Y_x, T)
        L_y = Squared_error_loss(Y, T)

        # replace Y by Y_x
        def expand(x):
            return substitute(x, [(Y, Y_x)])

        DY = join_rows([expand(gradient(L_y, Y.row(n))) for n in range(N)])

        for i in range(K):
            e_i = identity(K).col(i)
            for j in range(D):
                w_ij = W[i, j]
                e1 = gradient(L_x, w_ij)
                e2 = sum1((expand(gradient(L_y, Y.row(n))) * Y_x.row(n).jacobian([w_ij]) for n in range(N)))
                e3 = sum1((expand(gradient(L_y, Y.row(n))) * (X[n, j] * e_i) for n in range(N)))
                e4 = sum1(sum1(expand(gradient(L_y, Y[n, k])) * X[n, j] * e_i[k] for n in range(N)) for k in range(K))
                e5 = sum1(e_i[k] * sum1(expand(gradient(L_y, Y[n, k])) * X[n, j] for n in range(N)) for k in range(K))
                e6 = sum1(e_i[k] * (DY.T * X)[k, j] for k in range(K))
                e7 = (DY.T * X)[i, j]
                e1 = to_number(e1)
                e2 = to_number(e2)
                e3 = to_number(e3)
                e4 = to_number(e4)
                e5 = to_number(e5)
                self.assertTrue(e1.equals(e2))
                self.assertTrue(e2.equals(e3))
                self.assertTrue(e3.equals(e4))
                self.assertTrue(e4.equals(e5))
                self.assertTrue(e5.equals(e6))
                self.assertTrue(e6.equals(e7))


    def test_derivation_DW_new(self):
        N = 3
        D = 4
        K = 2

        X = matrix('X', N, D)
        W = matrix('D', K, D)
        Y = matrix('Y', N, K)
        b = matrix('b', 1, K)
        T = matrix('T', N, K)

        Y_x = X * W.T + ones(N) * b
        L_x = Squared_error_loss(Y_x, T)
        L_y = Squared_error_loss(Y, T)

        # replace Y by Y_x
        def expand(x):
            if isinstance(x, sp.MatrixBase):
                return substitute(x, [(Y, Y_x)])
            else:
                return to_number(substitute(to_matrix(x), [(Y, Y_x)]))

        DY = join_rows([expand(gradient(L_y, Y.row(n))) for n in range(N)])
        DW = DY.T * X

        for i in range(K):
            for j in range(D):
                e1 = sp.diff(L_x, W[i, j])
                e2 = sum1(expand(sp.diff(L_y, Y[n, k])) * sp.diff(Y_x[n, k], W[i, j]) for k in range(K) for n in range(N))
                self.assertTrue(e1.equals(e2))

        for n in range(N):
            for k in range(K):
                self.assertEqual(Y_x[n, k], sum1(X[n, d] * W[k, d] for d in range(D)) + b[k])

        for i in range(K):
            for j in range(D):
                for n in range(N):
                    for k in range(K):
                        e1 = sp.diff(Y_x[n, k], W[i, j])
                        e2 = X[n, j] if k == i else 0
                        self.assertTrue(e1.equals(e2))

        for i in range(K):
            for j in range(D):
                e1 = sp.diff(L_x, W[i, j])
                e2 = sum1(expand(sp.diff(L_y, Y[n, i])) * X[n, j] for n in range(N))
                e3 = DW[i, j]
                self.assertTrue(e1.equals(e2))
                self.assertTrue(e2.equals(e3))


class TestSoftmaxLayerDerivation(TestCase):
    def test_dL_dzi_derivation(self):
        N = 2
        D = 4
        K = 3

        x = matrix('x', N, D)
        y = matrix('y', N, K)
        w = matrix('w', K, D)
        z = matrix('z', N, K)

        b = matrix('b', 1, K)
        W = w

        # feedforward
        Z = x * w.T + row_repeat(b, N)
        Y = softmax(Z)

        L = lambda Y: to_matrix(squared_error_rows(Y))

        i = 1
        y_i = y.row(i)
        z_i = z.row(i)

        L_y = L(y)
        L_z = L(softmax(z))
        dL_dzi = substitute(L_z.jacobian(z_i), (z, Z))
        dL_dyi = substitute(L_y.jacobian(y_i), (y, Y))
        dsoftmax_dzi = substitute(softmax(z_i).jacobian(z_i), (z, Z))
        self.assertTrue(equal_matrices(dL_dzi, dL_dyi * dsoftmax_dzi))

        Dzi = dL_dzi
        Dyi = dL_dyi
        R1 = substitute(Dyi * (Diag(y_i) - y_i.T * y_i), (y, Y))
        R2 = substitute(hadamard(Dyi, y_i) - Dyi * y_i.T * y_i, (y, Y))
        self.assertTrue(equal_matrices(Dzi, R1))
        self.assertTrue(equal_matrices(R1, R2))

    def test_softmax_gradient(self):
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


class TestLogSoftmaxLayerDerivation(TestCase):
    def test_dL_dzi_derivation(self):
        N = 2
        D = 4
        K = 3

        x = matrix('x', N, D)
        y = matrix('y', N, K)
        w = matrix('w', K, D)
        z = matrix('z', N, K)

        b = matrix('b', 1, K)
        W = w

        # feedforward
        Z = x * w.T + row_repeat(b, N)
        Y = log_softmax(Z)

        L = lambda Y: to_matrix(squared_error_rows(Y))

        i = 1
        y_i = y.row(i)
        z_i = z.row(i)

        L_y = L(y)
        L_z = L(log_softmax(z))
        dL_dzi = substitute(L_z.jacobian(z_i), (z, Z))
        dL_dyi = substitute(L_y.jacobian(y_i), (y, Y))
        dlog_softmax_dzi = substitute(log_softmax(z_i).jacobian(z_i), (z, Z))
        self.assertTrue(equal_matrices(dL_dzi, dL_dyi * dlog_softmax_dzi))

        Dzi = dL_dzi
        Dyi = dL_dyi
        R = substitute(identity(K) - row_repeat(softmax(z_i), K), (z, Z))
        self.assertTrue(equal_matrices(dlog_softmax_dzi, R))
        self.assertTrue(equal_matrices(Dzi, Dyi * R))

    def test_log_softmax_gradient(self):
        K = 3
        N = 4
        Z = matrix('Z', N, K)
        Y = matrix('Y', N, K)
        T = matrix('T', N, K)
        L = Cross_entropy_loss

        # replace Y by Y_x
        def expand(x):
            return substitute(x, [(Y, log_softmax(Z))])

        for i in range(N):
            dL_dz_i = gradient(L(log_softmax(Z), T), Z.row(i))
            dL_dy_i = expand(gradient(L(Y, T), Y.row(i)))
            dlog_softmax_dz_i = log_softmax(Z.row(i)).jacobian(Z.row(i))
            self.assertTrue(dL_dz_i.equals(dL_dy_i * dlog_softmax_dz_i))

            e1 = dL_dz_i
            e2 = dL_dy_i * dlog_softmax_dz_i
            e3 = dL_dy_i * (identity(K) - ones(K) * softmax(Z.row(i)))
            self.assertTrue(e1.equals(e2))
            self.assertTrue(e2.equals(e3))

            Dz_i = dL_dz_i
            Dy_i = dL_dy_i
            f1 = Dz_i
            f2 = Dy_i * (identity(K) - ones(K) * softmax(Z.row(i)))
            f3 = Dy_i - Dy_i * ones(K) * softmax(Z.row(i))
            self.assertTrue(f1.equals(f2))
            self.assertTrue(f2.equals(f3))

        DY = expand(gradient(L(Y, T), Y))
        DZ = expand(DY - hadamard(softmax(Z), column_repeat(rows_sum(DY), K)))
        DZ1 = gradient(L(log_softmax(Z), T), Z)
        self.assertTrue(DZ.equals(DZ1))


class TestBatchNormDerivation(TestCase):
    def test_derivation_Dx(self):
        N = 2
        x = matrix('x', N, 1)
        r = matrix('r', N, 1)
        z = matrix('z', N, 1)

        I = identity(N) - ones(N, N) / N
        R = lambda r: I * x
        Sigma = lambda r: (r.T * r) / N
        Z = lambda r: to_number(inv_sqrt(Sigma(r))) * r
        L = lambda Y: to_matrix(squared_error_columns(Y))

        z_r = Z(r)
        dz_dr = z_r.jacobian(r)
        self.assertTrue(equal_matrices(dz_dr, to_number(inv_sqrt(Sigma(r)) / N) * (N * identity(N) - z_r * z_r.T)))

        L_r = L(z_r)
        L_z = L(z)
        dL_dr = L_r.jacobian(r)
        dL_dz = L_z.jacobian(z)
        Dr = dL_dr.T
        Dz = substitute(dL_dz.T, (z, z_r))
        self.assertTrue(equal_matrices(Dr, to_number(inv_sqrt(Sigma(r)) / N) * (N * identity(N) - z_r * z_r.T) * Dz))

        r_x = R(x)
        z_x = Z(r_x)
        L_x = L(z_x)
        Dx = L_x.jacobian(x).T
        Dr = substitute(Dr, (r, r_x))
        self.assertTrue(equal_matrices(Dx, I * Dr))

        sigma = Sigma(r_x)
        Dz = substitute(dL_dz.T, (z, z_x))
        z = z_x
        self.assertTrue(equal_matrices(Dx, to_number(inv_sqrt(sigma) / N) * (N * I * Dz - z * z.T * Dz), simplify_arguments=True))

    def test_derivation_DW(self):
        N = 2
        K = 2
        D = 3
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        L = lambda Y: to_matrix(squared_error_rows(Y))
        I = identity(N)
        i = 1
        j = 1
        w_j = w.row(j)
        x_i = x.row(i)
        e_j = I.col(j)

        Y = x * w.T + row_repeat(b, N)
        y_i = Y.row(i)

        Dw_j = L(y_i).jacobian(w_j)
        Dy_i = substitute(L(y).jacobian(y.row(i)), (y.row(i), y_i))
        dyi_dwj = y_i.jacobian(w.row(j))

        self.assertTrue(equal_matrices(dyi_dwj, e_j * x_i))
        self.assertTrue(equal_matrices(Dw_j, Dy_i * e_j * x_i))

        DW = Matrix([L(y_i).jacobian(w.row(j)) for j in range(K)])
        self.assertTrue(equal_matrices(DW, Dy_i.T * x_i))


if __name__ == '__main__':
    import unittest
    unittest.main()