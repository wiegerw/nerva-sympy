# nerva-sympy

[![PyPI](https://img.shields.io/pypi/v/nerva-sympy.svg)](https://pypi.org/project/nerva-sympy/)
[![License: BSL-1.0](https://img.shields.io/badge/license-BSL%201.0-blue.svg)](https://opensource.org/licenses/BSL-1.0)

**`nerva-sympy`** provides **symbolic validation** of multilayer perceptron implementations using **[SymPy](https://www.sympy.org/)**.  
It is part of the [Nerva](https://github.com/wiegerw/nerva) project — a suite of Python and C++ libraries that provide well-specified, inspectable implementations of neural networks.

➡️ **Unlike the other backends (`nerva-torch`, `nerva-numpy`, `nerva-jax`, `nerva-tensorflow`) which implement forward and backward passes, `nerva-sympy` is a *testing library***.  
It derives exact symbolic gradients and checks them against the manually implemented backpropagation code.

---

## 🗺️ Overview

The `nerva` libraries aim to make neural networks mathematically precise and transparent.  
While the implementation backends focus on execution, **`nerva-sympy` ensures correctness**:

- Provides **symbolic derivatives** of activation functions, layers, and loss functions.
- Validates **handwritten backpropagation equations** used in the other Nerva packages.
- Detects implementation errors early by comparing intermediate symbolic and numeric results.
- Avoids numerical gradient checking (finite differences) in favor of exact symbolic differentiation.

---

## 📦 Available Python Packages

Each backend has a dedicated PyPI package and GitHub repository:

| Package             | Backend     | PyPI                                               | GitHub                                                  |
|---------------------|-------------|----------------------------------------------------|----------------------------------------------------------|
| `nerva-jax`         | JAX         | [nerva-jax](https://pypi.org/project/nerva-jax/)           | [repo](https://github.com/wiegerw/nerva-jax)            |
| `nerva-numpy`       | NumPy       | [nerva-numpy](https://pypi.org/project/nerva-numpy/)       | [repo](https://github.com/wiegerw/nerva-numpy)          |
| `nerva-tensorflow`  | TensorFlow  | [nerva-tensorflow](https://pypi.org/project/nerva-tensorflow/) | [repo](https://github.com/wiegerw/nerva-tensorflow)     |
| `nerva-torch`       | PyTorch     | [nerva-torch](https://pypi.org/project/nerva-torch/)       | [repo](https://github.com/wiegerw/nerva-torch)          |
| `nerva-sympy`       | SymPy       | [nerva-sympy](https://pypi.org/project/nerva-sympy/)       | [repo](https://github.com/wiegerw/nerva-sympy)          |

> 📝 `nerva-sympy` itself only requires `numpy` and `sympy`. The other four packages are optional: they are only
> needed to compare their numerical implementations with the symbolic reference (see [Running Tests](#-running-tests)).

See the [nerva meta-repo](https://github.com/wiegerw/nerva) for an overview of all Python and C++ variants.

---

## 🚀 Quick Start

### Installation

The library can be installed in two ways: from the source repository or from the Python Package Index (PyPI).

```bash
# Install from the local repository
pip install .
```

```bash
# Install directly from PyPI
pip install nerva-sympy
```

The numerical backends can be installed as optional extras:

```bash
pip install "nerva-sympy[torch]"   # one backend: jax, numpy, tensorflow or torch
pip install "nerva-sympy[all]"     # all four backends
pip install "nerva-sympy[test]"    # pytest
```

### Example: Validate Softmax Backpropagation

This example validates the gradient computation of the **softmax layer**.  
The manually implemented backpropagation rules are checked against **symbolic differentiation**.

```python
# Backpropagation equations
DZ = hadamard(Y, DY - row_repeat(diag(Y.T * DY).T, K))
DW = DZ * X.T
Db = rows_sum(DZ)
DX = W.T * DZ

# Symbolic reference
DW1 = gradient(loss(Y), w)
Db1 = gradient(loss(Y), b)
DX1 = gradient(loss(Y), x)
DZ1 = gradient(loss(Y), z)

# Check equivalence
assert equal_matrices(DW, DW1)
assert equal_matrices(Db, Db1)
assert equal_matrices(DX, DX1)
assert equal_matrices(DZ, DZ1)
```

---

## 🧪 Running Tests

The test suite consists of three groups of tests, which need different packages:

| Tests | What they check | Requirements |
|-------|-----------------|--------------|
| Symbolic tests (`test_derivatives.py`, `test_layer_*.py`, `test_*_derivations.py`, ...) | The matrix-form equations against symbolic differentiation | `numpy`, `sympy` |
| Equation tests (`test_equations.py`) | The layer equations in the source code of each backend are identical to the validated SymPy equations | The source code of the backends (see below), no frameworks |
| Cross-framework tests (`test_matrix_operations.py`, `test_activation_functions.py`, `test_loss_functions.py`, `test_softmax_functions.py`, `test_one_hot.py`, `test_frameworks.py`) | The numerical results of the backends against SymPy and against each other | All four backends: `pip install "nerva-sympy[all]"` |

Test modules whose requirements are not installed are skipped, so the tests can be run with any subset of the backends.

The equation tests locate the source code of a backend in the following order:

1. The environment variable `NERVA_<BACKEND>_LAYERS_FILE`, e.g. `NERVA_TORCH_LAYERS_FILE=/path/to/layers.py`.
2. A sibling checkout, e.g. `../nerva-torch`.
3. An installed package, e.g. `nerva_torch`. The framework does not need to be installed (`pip install --no-deps nerva-torch`).

`test_frameworks.py` also needs the MNIST dataset in `data/mnist-flattened.npz`:

```bash
cd data
python prepare_data.py --dataset=mnist --download
```

To run the tests:

```bash
pip install ".[test]"           # add extras, e.g. ".[all,test]", to include the cross-framework tests
python tests/run_all_tests.py --run
```

The helper script uses pytest if it is installed, and falls back to unittest otherwise. It adds `src` to
`PYTHONPATH`, so the tests use the local source code. Additional arguments are passed to the test runner:

```bash
# Run only tests whose names match "softmax"
python tests/run_all_tests.py --run -k softmax

# Run a specific test file, class, or test case (pytest syntax)
python tests/run_all_tests.py --run tests/test_softmax_functions.py::TestSoftmax::test_softmax
```

The tests can also be run directly:

```bash
python -m pytest tests
python -m unittest discover -s tests -p "test_*.py" -v
```

Controls for output and verbosity:
- Set `NERVA_TEST_VERBOSE=1` to print the intermediate matrices and numbers of the tests.
- Set `NERVA_PYTEST_FLAGS` to override the pytest flags of the helper script, e.g. `NERVA_PYTEST_FLAGS="-ra -s"`.
- Set `NERVA_CHECK_VERBOSE=v` or `vv` for more output from the equation tests.
- Set `NERVA_TEST_SEED` to change the random seed of `test_frameworks.py` (default: 12345).

## 🧪 Validation Suite

The test suite covers **activation functions, layers, loss functions, and matrix operations**.  
Each test compares symbolic derivatives to the manually implemented backpropagation code.

Available tests:

- **Activation Functions**  
  `test_activation_functions.py`,
  `test_softmax_functions.py`, `test_softmax_function_derivations.py`  
  Validates symbolic gradients for Sigmoid, ReLU, SReLU, etc.

- **Layer Derivatives**  
  `test_layer_linear.py`, `test_layer_softmax.py`, `test_layer_batch_normalization.py`, `test_layer_dropout.py`, `test_layer_srelu.py`, `test_layer_derivations.py`  
  Checks symbolic vs. manual backpropagation for individual layers.

- **Loss Functions**  
  `test_loss_functions.py`, `test_loss_function_derivations.py` Ensures correct gradient formulas for common loss functions.

- **Supporting Operations**  
  `test_matrix_operations.py`, `test_matrix_properties.py`, `test_one_hot.py`, `test_derivatives.py`  
  Validates core symbolic building blocks.

- **Implementation Equations**  
  `test_equations.py`  
  Checks that the layer equations in the source code of the backends match the validated SymPy equations.

Note on the cross-framework consistency test (tests/test_frameworks.py):
- The test compares intermediates (X, Y, DY, T) across JAX, NumPy, TensorFlow and PyTorch while performing multiple SGD steps.
- It uses a custom pseudo-Huber loss function. Unlike the absolute error loss that was used before, its gradient is
  continuous, so tiny floating point differences between the frameworks cannot flip the sign of the gradient.
  The gradient is validated symbolically in `test_loss_function_derivations.py`.

---

## 🔢 Implementation Philosophy

Unlike frameworks that rely on autograd or approximate numerical checks:

- `nerva-sympy` uses **exact symbolic differentiation**.  
- This avoids floating-point instability and provides rigorous correctness guarantees.  
- All computations are expressed in **batch matrix form**, consistent with the other `nerva` libraries.  
- The test suite acts as a **ground truth oracle** for verifying implementations in JAX, NumPy, PyTorch, and TensorFlow.

---

## 📚 Relevant Papers

1. **Nerva: a Truly Sparse Implementation of Neural Networks**

   [arXiv:2407.17437](https://arxiv.org/abs/2407.17437)
   Introduces the library and reports sparse training experiments.

2. **Batch Matrix-form Equations and Implementation of Multilayer Perceptrons**

   [arXiv:2511.11918](https://arxiv.org/abs/2511.11918)
   Includes mathematical specifications and derivations.

---

## 📜 License

Distributed under the [Boost Software License 1.0](http://www.boost.org/LICENSE_1_0.txt).  
[License file](https://github.com/wiegerw/nerva-sympy/blob/main/LICENSE)

---

## 🙋 Contributing

Bug reports and contributions are welcome via the [GitHub issue tracker](https://github.com/wiegerw/nerva-sympy/issues).
