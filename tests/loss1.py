#!/usr/bin/env python3

from unittest import TestCase

import numpy as np
import torch
from nerva_numpy.matrix_operations import elements_sum, hadamard
from tests.utilities import to_torch


def Mean_squared_error_loss_torch(Y, T):
    loss = torch.nn.MSELoss(reduction='mean')  # reduction='mean' is the default
    return loss(Y, T)


def Squared_error_loss(Y, T):
    return elements_sum(hadamard(Y - T, Y - T))


def Mean_squared_error_loss(Y, T):
    N, K = Y.shape
    return Squared_error_loss(Y, T) / (K * N)


class TestLossFunctionValues(TestCase):
    def test_loss_function(self):
        Y = np.array([
            [1, 2, 3],
            [7, 3, 4]
        ], dtype=np.float32)

        T = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)

        loss_torch = Mean_squared_error_loss_torch(to_torch(Y), to_torch(T))
        loss_numpy = Mean_squared_error_loss(Y, T)
        print(loss_torch, loss_numpy)
        self.assertAlmostEqual(loss_torch, loss_numpy, delta=1e-10)


if __name__ == '__main__':
    import unittest
    unittest.main()
