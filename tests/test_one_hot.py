#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
from unittest import TestCase

import numpy as np
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def to_one_hot_numpy(x: np.ndarray, n_classes: int):
    one_hot = np.zeros((len(x), n_classes), dtype=float)
    one_hot[np.arange(len(x)), x] = 1
    return one_hot


def to_one_hot_torch(x: torch.LongTensor, n_classes: int):
    one_hot = torch.zeros(len(x), n_classes, dtype=torch.float)
    one_hot.scatter_(1, x.unsqueeze(1), 1)
    return one_hot


def to_one_hot_tensorflow(x: tf.Tensor, n_classes: int):
    one_hot = tf.one_hot(x, n_classes, dtype=tf.float64, axis=1)
    return one_hot


class TestOneHot(TestCase):
    def test_to_one_hot_numpy(self):
        x = np.array([1, 2, 0, 3, 4, 1])
        expected = np.array([[0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0]])
        result = to_one_hot_numpy(x, n_classes=5)
        np.testing.assert_array_equal(result, expected)

    def test_to_one_hot_torch(self):
        x = torch.LongTensor([1, 2, 0, 3, 4, 1])
        expected = torch.tensor([[0, 1, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [1, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 0]])
        result = to_one_hot_torch(x, n_classes=5)
        self.assertTrue(torch.all(torch.eq(result, expected)))

    def test_to_one_hot_tensorflow(self):
        x = tf.constant([1, 2, 0, 3, 4, 1])
        expected = tf.constant([[0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0]], dtype=tf.float64)
        result = to_one_hot_tensorflow(x, n_classes=5)
        self.assertTrue(tf.reduce_all(tf.equal(result, expected)))


if __name__ == '__main__':
    import unittest
    unittest.main()