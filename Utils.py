from random import uniform
from math import e


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
        weights = []
        for i in range(number_of_weights):
            weights.append(round(uniform(-1, 1), 2))
        return weights

    @staticmethod
    def node_bias_provider(): return 0

    @staticmethod
    def calculate_cost(desired, actual):
        if len(desired) != len(actual):
            return None
        cost = 0
        for i, desired in enumerate(desired):
            cost += (desired - actual[i])**2

        return cost
