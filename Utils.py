from random import uniform
from math import e
from functools import reduce
import numpy as np


class Utils:

    @staticmethod
    def sigmoid_function(x): return 1 / (1 + e**(-x))

    # TODO: figure out how to compute this based on the sigmoid_function used
    @staticmethod
    def sigmoid_derivative_function(x):
        y = Utils.sigmoid_function(x)
        return y * (1 - y)

    @staticmethod
    def node_weight_provider(number_of_weights):
        weights_ = [round(uniform(-1, 1), 2) for _ in range(number_of_weights)]
        return np.array(weights_)

    @staticmethod
    def node_bias_provider(): return 0

    @staticmethod
    def cost_function(desired, actual):
        if len(desired) != len(actual):
            return None
        pairs = zip(desired, actual)
        first = Utils.single_node_cost(pairs.__next__())
        return reduce(lambda x, y: x + Utils.single_node_cost(y), pairs, first)

    @staticmethod
    def single_node_cost(pair):
        delta = pair[0] - pair[1]
        return delta ** 2

    # TODO: figure out how to compute this based on the cost_function used
    @staticmethod
    def cost_derivative_function(desired, actual):
        return -2 * (desired - actual)

    @staticmethod
    def add_to_average(previous_average, denominator, new_value):
        return previous_average * ((denominator-1) / denominator) + new_value / denominator
