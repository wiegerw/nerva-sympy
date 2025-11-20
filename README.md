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

> 📝 `nerva-sympy` depends on the other four packages, since it validates their implementations.

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

Controls for output and verbosity
- Individual test names (default): The helper script runs pytest with -v, so you see each test as it runs.
- Suppress internal prints from tests: By default the test utilities do not print intermediate matrices/numbers. Set NERVA_TEST_VERBOSE=1 to enable those prints when needed.
- Override pytest flags: Set NERVA_PYTEST_FLAGS to customize, e.g., NERVA_PYTEST_FLAGS="-ra -s" ./tests/run_all_tests.sh to also show print output from successful tests.
- Unittest fallback: Uses -b (buffered) and -v by default.



To run the test suite locally:

1) Install dependencies (pytest is optional but recommended for nicer output):

```bash
pip install -r requirements.txt
pip install pytest  # optional
```

2) Run all tests via the helper script (it adds src to PYTHONPATH automatically):

```bash
./tests/run_all_tests.sh
```

You can pass additional arguments to the underlying test runner, for example:

```bash
# Run only tests whose names match "jacobian"
./tests/run_all_tests.sh -k "jacobian"

# Run a specific test file, class, or test case (pytest syntax)
./tests/run_all_tests.sh tests/test_softmax_functions.py::TestSoftmax::test_softmax
```

If you prefer to run without the script or are on a platform without bash:

```bash
# Using pytest
python3 -m pytest -s tests

# Or using unittest discovery
python3 -m unittest discover -s tests -p "test_*.py" -v
```

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
  `test_matrix_operations.py`, `test_one_hot.py`, `test_derivatives.py`, `test_lemmas.py`  
  Validates core symbolic building blocks.

Note on the cross-framework consistency test (tests/test_frameworks.py):
- The test compares intermediates (X, Y, DY, T) across JAX, NumPy, TensorFlow and PyTorch while performing multiple SGD steps.
- Currently, the test uses the AbsoluteError loss, which is not differentiable at 0. This sometimes causes the test to fail, even when all implementations are correct.

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
