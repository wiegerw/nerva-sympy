#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from unittest import TestCase

import numpy as np
from nerva_jax import activation_functions as jnp_
from nerva_numpy import activation_functions as np_
from nerva_sympy import activation_functions as sympy_
from nerva_sympy.activation_functions import *
from nerva_tensorflow import activation_functions as tf_
from nerva_torch import activation_functions as torch_

from utilities import check_arrays_equal, to_jax, to_numpy, to_sympy, to_tensorflow, to_torch


class TestActivationFunctions1D(TestCase):

    def test_relu(self):
        f = relu
        f1 = relu_derivative
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_leaky_relu(self):
        alpha = sp.symbols('alpha', real=True)
        f = leaky_relu(alpha)
        f1 = leaky_relu_derivative(alpha)
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_all_relu(self):
        alpha = sp.symbols('alpha', real=True)
        f = all_relu(alpha)
        f1 = all_relu_derivative(alpha)
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_hyperbolic_tangent(self):
        f = hyperbolic_tangent
        f1 = hyperbolic_tangent_derivative
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_sigmoid(self):
        f = sigmoid
        f1 = sigmoid_derivative
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))

    def test_srelu(self):
        al = sp.symbols('al', real=True)
        tl = sp.symbols('tl', real=True)
        ar = sp.symbols('ar', real=True)
        tr = sp.symbols('tr', real=True)

        f = srelu(al, tl, ar, tr)
        f1 = srelu_derivative(al, tl, ar, tr)
        x = sp.symbols('x', real=True)
        self.assertEqual(f1(x), f(x).diff(x))


class TestLogSigmoid(TestCase):

    def test_log_sigmoid(self):
        f = lambda x: sp.log(sigmoid(x))
        f1 = lambda x: 1 - sigmoid(x)
        x = sp.symbols('x', real=True)
        self.assertEqual(sp.simplify(f1(x)), sp.simplify(f(x).diff(x)))


if __name__ == '__main__':
    import unittest
    unittest.main()


class TestActivationFunctions(TestCase):
    def make_variables(self):
        X = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=np.float32)

        alpha = 0.25
        al = 0.2
        tl = 0.1
        ar = 0.7
        tr = 0.3

        return X, alpha, al, tl, ar, tr

    def test_relu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Relu(to_sympy(x))
        x2 = np_.Relu(to_numpy(x))
        x3 = tf_.Relu(to_tensorflow(x))
        x4 = torch_.Relu(to_torch(x))
        x5 = jnp_.Relu(to_jax(x))
        check_arrays_equal(self, 'Relu', [x1, x2, x3, x4, x5])

        x1 = sympy_.Relu_gradient(to_sympy(x))
        x2 = np_.Relu_gradient(to_numpy(x))
        x3 = tf_.Relu_gradient(to_tensorflow(x))
        x4 = torch_.Relu_gradient(to_torch(x))
        x5 = jnp_.Relu_gradient(to_jax(x))
        check_arrays_equal(self, 'Relu_gradient', [x1, x2, x3, x4, x5])

    def test_leaky_relu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Leaky_relu(alpha)(to_sympy(x))
        x2 = np_.Leaky_relu(alpha)(to_numpy(x))
        x3 = tf_.Leaky_relu(alpha)(to_tensorflow(x))
        x4 = torch_.Leaky_relu(alpha)(to_torch(x))
        x5 = jnp_.Leaky_relu(alpha)(to_jax(x))
        check_arrays_equal(self, 'Leaky_relu', [x1, x2, x3, x4, x5])

        x1 = sympy_.Leaky_relu_gradient(alpha)(to_sympy(x))
        x2 = np_.Leaky_relu_gradient(alpha)(to_numpy(x))
        x3 = tf_.Leaky_relu_gradient(alpha)(to_tensorflow(x))
        x4 = torch_.Leaky_relu_gradient(alpha)(to_torch(x))
        x5 = jnp_.Leaky_relu_gradient(alpha)(to_jax(x))
        check_arrays_equal(self, 'Leaky_relu_gradient', [x1, x2, x3, x4, x5])

    def test_All_relu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.All_relu(alpha)(to_sympy(x))
        x2 = np_.All_relu(alpha)(to_numpy(x))
        x3 = tf_.All_relu(alpha)(to_tensorflow(x))
        x4 = torch_.All_relu(alpha)(to_torch(x))
        x5 = jnp_.All_relu(alpha)(to_jax(x))
        check_arrays_equal(self, 'All_relu', [x1, x2, x3, x4, x5])

        x1 = sympy_.All_relu_gradient(alpha)(to_sympy(x))
        x2 = np_.All_relu_gradient(alpha)(to_numpy(x))
        x3 = tf_.All_relu_gradient(alpha)(to_tensorflow(x))
        x4 = torch_.All_relu_gradient(alpha)(to_torch(x))
        x5 = jnp_.All_relu_gradient(alpha)(to_jax(x))
        check_arrays_equal(self, 'All_relu_gradient', [x1, x2, x3, x4, x5])

    def test_Hyperbolic_tangent(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Hyperbolic_tangent(to_sympy(x))
        x2 = np_.Hyperbolic_tangent(to_numpy(x))
        x3 = tf_.Hyperbolic_tangent(to_tensorflow(x))
        x4 = torch_.Hyperbolic_tangent(to_torch(x))
        x5 = jnp_.Hyperbolic_tangent(to_jax(x))
        check_arrays_equal(self, 'Hyperbolic_tangent', [x1, x2, x3, x4, x5])

        x1 = sympy_.Hyperbolic_tangent_gradient(to_sympy(x))
        x2 = np_.Hyperbolic_tangent_gradient(to_numpy(x))
        x3 = tf_.Hyperbolic_tangent_gradient(to_tensorflow(x))
        x4 = torch_.Hyperbolic_tangent_gradient(to_torch(x))
        x5 = jnp_.Hyperbolic_tangent_gradient(to_jax(x))
        check_arrays_equal(self, 'Hyperbolic_tangent_gradient', [x1, x2, x3, x4, x5])

    def test_Sigmoid(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Sigmoid(to_sympy(x))
        x2 = np_.Sigmoid(to_numpy(x))
        x3 = tf_.Sigmoid(to_tensorflow(x))
        x4 = torch_.Sigmoid(to_torch(x))
        x5 = jnp_.Sigmoid(to_jax(x))
        check_arrays_equal(self, 'Sigmoid', [x1, x2, x3, x4, x5])

        x1 = sympy_.Sigmoid_gradient(to_sympy(x))
        x2 = np_.Sigmoid_gradient(to_numpy(x))
        x3 = tf_.Sigmoid_gradient(to_tensorflow(x))
        x4 = torch_.Sigmoid_gradient(to_torch(x))
        x5 = jnp_.Sigmoid_gradient(to_jax(x))
        check_arrays_equal(self, 'Sigmoid_gradient', [x1, x2, x3, x4, x5])

    def test_Srelu(self):
        X, alpha, al, tl, ar, tr = self.make_variables()

        x = X
        x1 = sympy_.Srelu(al, tl, ar, tr)(to_sympy(x))
        x2 = np_.Srelu(al, tl, ar, tr)(to_numpy(x))
        x3 = tf_.Srelu(al, tl, ar, tr)(to_tensorflow(x))
        x4 = torch_.Srelu(al, tl, ar, tr)(to_torch(x))
        x5 = jnp_.Srelu(al, tl, ar, tr)(to_jax(x))
        check_arrays_equal(self, 'Srelu', [x1, x2, x3, x4, x5])

        x1 = sympy_.Srelu_gradient(al, tl, ar, tr)(to_sympy(x))
        x2 = np_.Srelu_gradient(al, tl, ar, tr)(to_numpy(x))
        x3 = tf_.Srelu_gradient(al, tl, ar, tr)(to_tensorflow(x))
        x4 = torch_.Srelu_gradient(al, tl, ar, tr)(to_torch(x))
        x5 = jnp_.Srelu_gradient(al, tl, ar, tr)(to_jax(x))
        check_arrays_equal(self, 'Srelu_gradient', [x1, x2, x3, x4, x5])