# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import random
import tempfile

def construct_mlp_jax(datapath: str):
    import jax.numpy as jnp
    import numpy as np
    from nerva_jax.activation_functions import ActivationFunction, HyperbolicTangentActivation
    from nerva_jax.datasets import create_npz_dataloaders
    from nerva_jax.layers import ActivationLayer, LinearLayer
    from nerva_jax.loss_functions import LossFunction
    from nerva_jax.matrix_operations import elements_sum, Matrix
    from nerva_jax.multilayer_perceptron import MultilayerPerceptron
    from nerva_jax.optimizers import MomentumOptimizer, NesterovOptimizer, CompositeOptimizer
    from nerva_jax.weight_initializers import zero_bias, xavier_normalized_weights

    # ------------------------
    # Custom activation function
    # ------------------------

    def Elu(alpha):
        return lambda X: jnp.where(X > 0, X, alpha * (jnp.exp(X) - 1))

    def Elu_gradient(alpha):
        return lambda X: jnp.where(X > 0, jnp.ones_like(X), alpha * jnp.exp(X))

    class ELUActivation(ActivationFunction):
        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def __call__(self, X: Matrix) -> Matrix:
            return Elu(self.alpha)(X)

        def gradient(self, X: Matrix) -> Matrix:
            return Elu_gradient(self.alpha)(X)

    # ------------------------
    # Custom weight initializer
    # ------------------------

    def lecun_weights(W: Matrix) -> Matrix:
        K, D = W.shape
        stddev = jnp.sqrt(1.0 / D)
        return np.random.randn(K, D) * stddev

    # ------------------------
    # Custom loss function
    # ------------------------

    class AbsoluteErrorLossFunction(LossFunction):
        def __call__(self, Y: Matrix, T: Matrix) -> float:
            return elements_sum(abs(Y - T))

        def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
            return jnp.sign(Y - T)

    def create_model():
        M = MultilayerPerceptron()

        # configure layer 1
        layer1 = ActivationLayer(784, 1024, ELUActivation(0.1))
        xavier_normalized_weights(layer1.W)
        zero_bias(layer1.b)
        optimizer_W = MomentumOptimizer(layer1, "W", "DW", 0.9)
        optimizer_b = NesterovOptimizer(layer1, "b", "Db", 0.75)
        layer1.optimizer = CompositeOptimizer([optimizer_W, optimizer_b])

        # configure layer 2
        layer2 = ActivationLayer(1024, 512, HyperbolicTangentActivation())
        layer2.W = lecun_weights(layer2.W)
        layer2.b = zero_bias(layer2.b)
        layer2.set_optimizer("Momentum(0.8)")

        # configure layer 3
        layer3 = LinearLayer(512, 10)
        layer3.set_weights("He")
        layer3.set_optimizer("GradientDescent")

        M.layers = [layer1, layer2, layer3]
        return M

    M = create_model()
    train_loader, test_loader = create_npz_dataloaders(datapath, batch_size=100)
    loss: LossFunction = AbsoluteErrorLossFunction()
    return M, loss, train_loader


def construct_mlp_numpy(datapath: str):
    import numpy as np
    from nerva_numpy.activation_functions import ActivationFunction, HyperbolicTangentActivation
    from nerva_numpy.datasets import create_npz_dataloaders
    from nerva_numpy.layers import ActivationLayer, LinearLayer
    from nerva_numpy.loss_functions import LossFunction
    from nerva_numpy.matrix_operations import elements_sum, Matrix
    from nerva_numpy.multilayer_perceptron import MultilayerPerceptron
    from nerva_numpy.optimizers import MomentumOptimizer, NesterovOptimizer, CompositeOptimizer
    from nerva_numpy.weight_initializers import set_bias_to_zero, set_weights_xavier_normalized

    # ------------------------
    # Custom activation function
    # ------------------------

    def Elu(alpha):
        return lambda X: np.where(X > 0, X, alpha * (np.exp(X) - 1))

    def Elu_gradient(alpha):
        return lambda X: np.where(X > 0, np.ones_like(X), alpha * np.exp(X))

    class ELUActivation(ActivationFunction):
        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def __call__(self, X: Matrix) -> Matrix:
            return Elu(self.alpha)(X)

        def gradient(self, X: Matrix) -> Matrix:
            return Elu_gradient(self.alpha)(X)

    # ------------------------
    # Custom weight initializer
    # ------------------------

    def set_weights_lecun(W: Matrix):
        K, D = W.shape
        stddev = np.sqrt(1.0 / D)
        W[:] = np.random.randn(K, D) * stddev

    # ------------------------
    # Custom loss function
    # ------------------------

    class AbsoluteErrorLossFunction(LossFunction):
        def __call__(self, Y: Matrix, T: Matrix) -> float:
            return elements_sum(abs(Y - T))

        def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
            return np.sign(Y - T)

    def create_model():
        M = MultilayerPerceptron()

        # configure layer 1
        layer1 = ActivationLayer(784, 1024, ELUActivation(0.1))
        set_weights_xavier_normalized(layer1.W)
        set_bias_to_zero(layer1.b)
        optimizer_W = MomentumOptimizer(layer1.W, layer1.DW, 0.9)
        optimizer_b = NesterovOptimizer(layer1.b, layer1.Db, 0.75)
        layer1.optimizer = CompositeOptimizer([optimizer_W, optimizer_b])

        # configure layer 2
        layer2 = ActivationLayer(1024, 512, HyperbolicTangentActivation())
        set_weights_lecun(layer2.W)
        set_bias_to_zero(layer2.b)
        layer2.set_optimizer("Momentum(0.8)")

        # configure layer 3
        layer3 = LinearLayer(512, 10)
        layer3.set_weights("He")
        layer3.set_optimizer("GradientDescent")

        M.layers = [layer1, layer2, layer3]
        return M

    M = create_model()
    train_loader, test_loader = create_npz_dataloaders(datapath, batch_size=100)
    loss: LossFunction = AbsoluteErrorLossFunction()
    return M, loss, train_loader


def construct_mlp_tensorflow(datapath: str):
    import tensorflow as tf
    from nerva_tensorflow.activation_functions import ActivationFunction, HyperbolicTangentActivation, ReLUActivation
    from nerva_tensorflow.datasets import create_npz_dataloaders
    from nerva_tensorflow.layers import ActivationLayer, LinearLayer
    from nerva_tensorflow.loss_functions import LossFunction
    from nerva_tensorflow.matrix_operations import Matrix
    from nerva_tensorflow.multilayer_perceptron import MultilayerPerceptron
    from nerva_tensorflow.optimizers import MomentumOptimizer, NesterovOptimizer, CompositeOptimizer
    from nerva_tensorflow.weight_initializers import set_bias_to_zero, set_weights_xavier_normalized

    # ------------------------
    # Custom activation function
    # ------------------------

    def Elu(alpha):
        return lambda X: tf.where(X > 0, X, alpha * (tf.exp(X) - 1))

    def Elu_gradient(alpha):
        return lambda X: tf.where(X > 0, tf.ones_like(X), alpha * tf.exp(X))

    class ELUActivation(ActivationFunction):
        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def __call__(self, X: Matrix) -> Matrix:
            return Elu(self.alpha)(X)

        def gradient(self, X: Matrix) -> Matrix:
            return Elu_gradient(self.alpha)(X)

    # ------------------------
    # Custom weight initializer
    # ------------------------

    def set_weights_lecun(W: Matrix):
        K, D = W.shape
        stddev = tf.sqrt(tf.constant(1.0 / D, dtype=tf.float32))
        W.assign(tf.random.normal((K, D), stddev=stddev, dtype=W.dtype))

    # ------------------------
    # Custom loss function
    # ------------------------

    class AbsoluteErrorLossFunction(LossFunction):
        def __call__(self, Y: Matrix, T: Matrix) -> float:
            return tf.reduce_sum(tf.abs(Y - T)).numpy()

        def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
            return tf.sign(Y - T)

    def create_model():
        M = MultilayerPerceptron()

        # configure layer 1
        layer1 = ActivationLayer(784, 1024, ELUActivation(0.1))
        set_weights_xavier_normalized(layer1.W)
        set_bias_to_zero(layer1.b)
        optimizer_W = MomentumOptimizer(layer1.W, layer1.DW, 0.9)
        optimizer_b = NesterovOptimizer(layer1.b, layer1.Db, 0.75)
        layer1.optimizer = CompositeOptimizer([optimizer_W, optimizer_b])

        # configure layer 2
        layer2 = ActivationLayer(1024, 512, HyperbolicTangentActivation())
        set_weights_lecun(layer2.W)
        set_bias_to_zero(layer2.b)
        layer2.set_optimizer("Momentum(0.8)")

        # configure layer 3
        layer3 = LinearLayer(512, 10)
        layer3.set_weights("He")
        layer3.set_optimizer("GradientDescent")

        M.layers = [layer1, layer2, layer3]
        return M

    M = create_model()
    train_loader, test_loader = create_npz_dataloaders(datapath, batch_size=100)
    loss: LossFunction = AbsoluteErrorLossFunction()
    return M, loss, train_loader


def construct_mlp_torch(datapath: str):
    import torch
    from nerva_torch.activation_functions import ActivationFunction, HyperbolicTangentActivation, ReLUActivation
    from nerva_torch.datasets import create_npz_dataloaders
    from nerva_torch.layers import ActivationLayer, LinearLayer
    from nerva_torch.loss_functions import LossFunction
    from nerva_torch.matrix_operations import elements_sum, Matrix
    from nerva_torch.multilayer_perceptron import MultilayerPerceptron
    from nerva_torch.optimizers import MomentumOptimizer, NesterovOptimizer, CompositeOptimizer
    from nerva_torch.weight_initializers import set_bias_to_zero, set_weights_xavier_normalized

    # ------------------------
    # Custom activation function
    # ------------------------

    def Elu(alpha):
        return lambda X: torch.where(X > 0, X, alpha * (torch.exp(X) - 1))

    def Elu_gradient(alpha):
        return lambda X: torch.where(X > 0, torch.ones_like(X), alpha * torch.exp(X))

    class ELUActivation(ActivationFunction):
        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def __call__(self, X: Matrix) -> Matrix:
            return Elu(self.alpha)(X)

        def gradient(self, X: Matrix) -> Matrix:
            return Elu_gradient(self.alpha)(X)

    # ------------------------
    # Custom weight initializer
    # ------------------------

    def set_weights_lecun(W: Matrix):
        K, D = W.shape
        stddev = torch.sqrt(torch.tensor(1.0 / D))
        W.data = torch.randn(K, D) * stddev

    # ------------------------
    # Custom loss function
    # ------------------------

    class AbsoluteErrorLossFunction(LossFunction):
        def __call__(self, Y: Matrix, T: Matrix) -> float:
            return elements_sum(abs(Y - T))

        def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
            return torch.sign(Y - T)

    def create_model():
        M = MultilayerPerceptron()

        # configure layer 1
        layer1 = ActivationLayer(784, 1024, ELUActivation(0.1))
        set_weights_xavier_normalized(layer1.W)
        set_bias_to_zero(layer1.b)
        optimizer_W = MomentumOptimizer(layer1.W, layer1.DW, 0.9)
        optimizer_b = NesterovOptimizer(layer1.b, layer1.Db, 0.75)
        layer1.optimizer = CompositeOptimizer([optimizer_W, optimizer_b])

        # configure layer 2
        layer2 = ActivationLayer(1024, 512, HyperbolicTangentActivation())
        set_weights_lecun(layer2.W)
        set_bias_to_zero(layer2.b)
        layer2.set_optimizer("Momentum(0.8)")

        # configure layer 3
        layer3 = LinearLayer(512, 10)
        layer3.set_weights("He")
        layer3.set_optimizer("GradientDescent")

        M.layers = [layer1, layer2, layer3]
        return M

    M = create_model()
    train_loader, test_loader = create_npz_dataloaders(datapath, batch_size=100)
    loss: LossFunction = AbsoluteErrorLossFunction()
    return M, loss, train_loader


def construct_models(data_path: str, synchronize_weights: bool = False):
    """
    Construct models for all four frameworks: jax, numpy, tensorflow, torch.

    Args:
        data_path: path to the dataset .npz file
        synchronize_weights: if True, copy weights from the torch model
                             into all other frameworks.

    Returns:
        dict mapping framework name -> (M, loss, loader)
    """
    constructors = {
        "jax": construct_mlp_jax,
        "numpy": construct_mlp_numpy,
        "tensorflow": construct_mlp_tensorflow,
        "torch": construct_mlp_torch,
    }

    models = {name: ctor(data_path) for name, ctor in constructors.items()}

    if synchronize_weights:
        # Pick a random framework to be the source
        source_name = random.choice(list(models.keys()))
        print(f'Copying weights from nerva_{source_name}')
        M_source, _, _ = models[source_name]

        # Create a temporary file path for weights
        fd, weights_file = tempfile.mkstemp(suffix=".npz")
        os.close(fd)  # Close the low-level file descriptor immediately

        try:
            # Save weights from the selected source framework
            M_source.save_weights_and_bias(weights_file)

            # Load into all other frameworks
            for name, (M, _, _) in models.items():
                if name != source_name:
                    M.load_weights_and_bias(weights_file)

        finally:
            if os.path.exists(weights_file):
                os.remove(weights_file)

    return models
