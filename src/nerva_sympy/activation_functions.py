# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import sympy as sp
from nerva_sympy.utilities import parse_function_call
from sympy import Lambda, Piecewise

Matrix = sp.Matrix

# Naming conventions:
# - lowercase functions operate on real numbers
# - uppercase functions operate on matrices


def relu(x):
    # return max(0, x)
    return Piecewise((0, x < 0), (x, True))


def relu_derivative(x):
    # return 0 if x < 0 else 1
    return Piecewise((0, x < 0), (1, True))


def leaky_relu(alpha):
    x = sp.symbols('x')
    # fx = max(alpha * x, x)
    fx = Piecewise((alpha * x, x < alpha * x), (x, True))
    return Lambda(x, fx)


def leaky_relu_derivative(alpha):
    x = sp.symbols('x')
    # fx = alpha if x < alpha * x else 1
    fx = Piecewise((alpha, x < alpha * x), (1, True))
    return Lambda(x, fx)


def all_relu(alpha):
    x = sp.symbols('x')
    # fx = alpha * x if x < 0 else x
    fx = Piecewise((alpha * x, x < 0), (x, True))
    return Lambda(x, fx)


def all_relu_derivative(alpha):
    x = sp.symbols('x')
    # fx = alpha if x < 0 else 1
    fx = Piecewise((alpha, x < 0), (1, True))
    return Lambda(x, fx)


def hyperbolic_tangent(x):
    return sp.tanh(x)


def hyperbolic_tangent_derivative(x):
    y = hyperbolic_tangent(x)
    return 1 - y * y


def sigmoid(x):
    return 1 / (1 + sp.exp(-x))


def sigmoid_derivative(x):
    y = sigmoid(x)
    return y * (1 - y)


def srelu(al, tl, ar, tr):
    x = sp.symbols('x')
    return Lambda(x, Piecewise((tl + al * (x - tl), x <= tl), (x, x < tr), (tr + ar * (x - tr), True)))


def srelu_derivative(al, tl, ar, tr):
    x = sp.symbols('x')
    return Lambda(x, Piecewise((al, x <= tl), (1, x < tr), (ar, True)))


def Relu(X: Matrix) -> Matrix:
    return X.applyfunc(relu)


def Relu_gradient(X: Matrix) -> Matrix:
    return X.applyfunc(relu_derivative)


def Leaky_relu(alpha):
    return lambda x: x.applyfunc(leaky_relu(alpha))


def Leaky_relu_gradient(alpha):
    return lambda x: x.applyfunc(leaky_relu_derivative(alpha))


def All_relu(alpha):
    return lambda x: x.applyfunc(all_relu(alpha))


def All_relu_gradient(alpha):
    return lambda x: x.applyfunc(all_relu_derivative(alpha))


def Hyperbolic_tangent(X: Matrix) -> Matrix:
    return X.applyfunc(hyperbolic_tangent)


def Hyperbolic_tangent_gradient(X: Matrix) -> Matrix:
    return X.applyfunc(hyperbolic_tangent_derivative)


def Sigmoid(X: Matrix) -> Matrix:
    return X.applyfunc(sigmoid)


def Sigmoid_gradient(X: Matrix) -> Matrix:
    return X.applyfunc(sigmoid_derivative)


def Srelu(al, tl, ar, tr):
    return lambda x: x.applyfunc(srelu(al, tl, ar, tr))


def Srelu_gradient(al, tl, ar, tr):
    return lambda x: x.applyfunc(srelu_derivative(al, tl, ar, tr))


class ActivationFunction(object):
    """Interface for activation functions with value and gradient methods."""
    def __call__(self, X: Matrix) -> Matrix:
        raise NotImplementedError

    def gradient(self, X: Matrix) -> Matrix:
        raise NotImplementedError


class ReLUActivation(ActivationFunction):
    """ReLU activation function: max(0, x)."""
    def __call__(self, X: Matrix) -> Matrix:
        return Relu(X)

    def gradient(self, X: Matrix) -> Matrix:
        """Compute gradient of ReLU."""
        return Relu_gradient(X)

    def __repr__(self) -> str:
        return "ReLU"

    __str__ = __repr__


class LeakyReLUActivation(ActivationFunction):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, X: Matrix) -> Matrix:
        """Apply leaky ReLU activation."""
        return Leaky_relu(self.alpha)(X)

    def gradient(self, X: Matrix) -> Matrix:
        """Compute gradient of leaky ReLU."""
        return Leaky_relu_gradient(self.alpha)(X)

    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={float(self.alpha)})"

    __str__ = __repr__


class AllReLUActivation(ActivationFunction):
    """AllReLU activation (alternative parameterization of leaky ReLU)."""
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, X: Matrix) -> Matrix:
        """Apply AllReLU activation."""
        return All_relu(self.alpha)(X)

    def gradient(self, X: Matrix) -> Matrix:
        """Compute gradient of AllReLU."""
        return All_relu_gradient(self.alpha)(X)

    def __repr__(self) -> str:
        return f"AllReLU(alpha={float(self.alpha)})"

    __str__ = __repr__


class HyperbolicTangentActivation(ActivationFunction):
    """Hyperbolic tangent activation function."""
    def __call__(self, X: Matrix) -> Matrix:
        return Hyperbolic_tangent(X)

    def gradient(self, X: Matrix) -> Matrix:
        """Compute gradient of hyperbolic tangent."""
        return Hyperbolic_tangent_gradient(X)

    def __repr__(self) -> str:
        return "HyperbolicTangent"

    __str__ = __repr__


class SigmoidActivation(ActivationFunction):
    """Sigmoid activation function: 1 / (1 + exp(-x))."""
    def __call__(self, X: Matrix) -> Matrix:
        return Sigmoid(X)

    def gradient(self, X: Matrix) -> Matrix:
        """Compute gradient of sigmoid."""
        return Sigmoid_gradient(X)

    def __repr__(self) -> str:
        return "Sigmoid"

    __str__ = __repr__


class SReLUActivation(ActivationFunction):
    """Smooth rectified linear activation with learnable parameters."""
    def __init__(self, al=0, tl=0, ar=0, tr=1):
        # Store the parameters and their gradients in matrices.
        # This is to make them usable for optimizers.
        self.x = Matrix([al, tl, ar, tr])
        self.Dx = Matrix([0, 0, 0, 0])

    def __call__(self, X: Matrix) -> Matrix:
        """Apply SReLU activation with current parameters."""
        al, tl, ar, tr = self.x
        return Srelu(al, tl, ar, tr)(X)

    def gradient(self, X: Matrix) -> Matrix:
        """Compute gradient of SReLU with current parameters."""
        al, tl, ar, tr = self.x
        return Srelu_gradient(al, tl, ar, tr)(X)

    def __repr__(self) -> str:
        al, tl, ar, tr = [float(v) for v in self.x]
        return f"SReLU(al={al}, tl={tl}, ar={ar}, tr={tr})"

    __str__ = __repr__


def parse_activation(text: str) -> ActivationFunction:
    """Parse a textual activation specification into an ActivationFunction.

    Examples include "ReLU", "Sigmoid", "HyperbolicTangent",
    "AllReLU(alpha=0.1)", "LeakyReLU(alpha=0.1)", and
    "SReLU(al=0, tl=0, ar=0, tr=1)".
    """
    try:
        func = parse_function_call(text)
        if func.name == 'ReLU':
            return ReLUActivation()
        elif func.name == 'Sigmoid':
            return SigmoidActivation()
        elif func.name == 'HyperbolicTangent':
            return HyperbolicTangentActivation()
        elif func.name == 'AllReLU':
            alpha = func.as_scalar('alpha')
            return AllReLUActivation(alpha)
        elif func.name == 'LeakyReLU':
            alpha = func.as_scalar('alpha')
            return LeakyReLUActivation(alpha)
        elif func.name == 'SReLU':
            al = func.as_scalar('al', 0)
            tl = func.as_scalar('tl', 0)
            ar = func.as_scalar('ar', 0)
            tr = func.as_scalar('tr', 1)
            return SReLUActivation(al, tl, ar, tr)
    except:
        pass
    raise RuntimeError(f'Could not parse activation "{text}"')
