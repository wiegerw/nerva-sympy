#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

from nerva_sympy.activation_functions import *
from nerva_sympy.matrix_operations import *

from utilities import equal_matrices, matrix


class TestSReLULayers(TestCase):

    def _test_srelu_layer(self, D, K, N, loss):
        # variables
        x = matrix('x', N, D)
        y = matrix('y', N, K)
        z = matrix('z', N, K)
        w = matrix('w', K, D)
        b = matrix('b', 1, K)
        al = sp.symbols('al', real=True)
        tl = sp.symbols('tl', real=True)
        ar = sp.symbols('ar', real=True)
        tr = sp.symbols('tr', real=True)
        X = x
        W = w

        act = SReLUActivation(al, tl, ar, tr)

        # helper functions
        Zij = sp.symbols('Zij')
        Al = lambda Z: Z.applyfunc(lambda Zij: Piecewise((Zij - tl, Zij <= tl), (0, True)))
        Ar = lambda Z: Z.applyfunc(lambda Zij: Piecewise((0, Zij <= tl), (0, Zij < tr), (Zij - tr, True)))
        Tl = lambda Z: Z.applyfunc(lambda Zij: Piecewise((1 - al, Zij <= tl), (0, True)))
        Tr = lambda Z: Z.applyfunc(lambda Zij: Piecewise((0, Zij <= tl), (0, Zij < tr), (1 - ar, True)))

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
        Dal = elements_sum(hadamard(DY, Al(Z)))
        Dar = elements_sum(hadamard(DY, Ar(Z)))
        Dtl = elements_sum(hadamard(DY, Tl(Z)))
        Dtr = elements_sum(hadamard(DY, Tr(Z)))

        # test gradients
        DZ1 = substitute(gradient(loss(act(z)), z), (z, Z))
        DW1 = gradient(loss(Y), w)
        Db1 = gradient(loss(Y), b)
        DX1 = gradient(loss(Y), x)
        self.assertTrue(equal_matrices(DZ, DZ1))
        self.assertTrue(equal_matrices(DW, DW1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Db, Db1, simplify_arguments=True))
        self.assertTrue(equal_matrices(DX, DX1, simplify_arguments=True))

        Dal1 = gradient(loss(Y), Matrix([[al]]))
        Dtl1 = gradient(loss(Y), Matrix([[tl]]))
        Dar1 = gradient(loss(Y), Matrix([[ar]]))
        Dtr1 = gradient(loss(Y), Matrix([[tr]]))
        # wrap values in a matrix to make them usable for the equal_matrices function
        Dal = Matrix([[Dal]])
        Dar = Matrix([[Dar]])
        Dtl = Matrix([[Dtl]])
        Dtr = Matrix([[Dtr]])
        self.assertTrue(equal_matrices(Dal, Dal1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Dtl, Dtl1, simplify_arguments=True))
        self.assertTrue(equal_matrices(Dar, Dar1, simplify_arguments=False))
        self.assertTrue(equal_matrices(Dtr, Dtr1, simplify_arguments=True))

    def test_srelu_layer(self):
        self._test_srelu_layer(D=3, K=2, N=2, loss = elements_sum)
        self._test_srelu_layer(D=2, K=3, N=2, loss = elements_sum)
        self._test_srelu_layer(D=2, K=2, N=3, loss = elements_sum)


if __name__ == '__main__':
    import unittest
    unittest.main()