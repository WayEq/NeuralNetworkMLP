from functools import reduce
from math import e, log
from random import uniform, shuffle

import numpy as np


def sigmoid_function(x): return 1 / (1 + e**(-x))


# TODO: figure out how to compute this based on the sigmoid_function used
def sigmoid_derivative_function(x):
    y = sigmoid_function(x)
    return y * (1 - y)


def node_weight_provider(number_of_weights):
    weights_ = [round(uniform(-1, 1), 2) for _ in range(number_of_weights)]
    return np.array(weights_)


def node_bias_provider(): return 0.0


def quadratic_cost_function(desired_vector, actual_vector):
    if len(desired_vector) != len(actual_vector):
        return None
    pairs = zip(desired_vector, actual_vector)
    first_pair = pairs.__next__()
    first = single_node_quadratic_cost(first_pair[0], first_pair[1])
    return reduce(lambda x, y: x + single_node_quadratic_cost(y[0], y[1]), pairs, first)


def calculate_l2_cost(weights, training_data_size, regularization_term):

    squares = np.square(weights)
    accumulator = number_of_weights = 0
    for i in squares:
        accumulator += np.sum(i)
        number_of_weights += i.shape[0] * i.shape[1]
    return accumulator * regularization_term / (2 * training_data_size)

def batch_cross_entropy_cost_function(desired_vector, actual_vector):
    batch_length = len(actual_vector)
    if len(desired_vector) != batch_length:
        return None
    vectorized = vectorized_result(desired_vector, len(desired_vector))
    summation = np.vectorize(np.vectorize(single_node_cross_entropy_cost))(vectorized, actual_vector)
    cost = summation.sum() * -1 / batch_length
    return cost


def batch_cross_entropy_cost_function_with_l2_regularize(desired_vector, actual_vector, weights, training_data_size, regularization_term=None):
    return batch_cross_entropy_cost_function(desired_vector,actual_vector) + calculate_l2_cost(weights, training_data_size, regularization_term)


def single_node_quadratic_cost(desired_value, actual_value):
    delta = desired_value - actual_value
    return delta ** 2


def single_node_cross_entropy_cost(desired_value, actual_value):
    return desired_value * log(actual_value) + (1-desired_value) * log(1-actual_value)


# TODO: figure out how to compute this based on the cost_function used
def quadratic_cost_derivative_function(desired, actual):
    return -2 * (desired - actual)


def quadratic_cost_error(desired, actual):
    return actual - desired


def add_to_average(previous_average, denominator, new_value):
    return previous_average * ((denominator-1) / denominator) + new_value / denominator


def cross_entropy_output_layer_calculator(node_activations, _, desired_output):
    errors = []
    for i, activation in enumerate(node_activations):
        error = quadratic_cost_error(desired_output[i], node_activations[i])
        errors.append(error)
    return errors


def output_layer_error_calculator(node_activations, node_z_values, desired_output):
    errors = []
    for i, z_value in enumerate(node_z_values):
        cost_derived = quadratic_cost_derivative_function(desired_output[i], node_activations[i])
        activation_derived = sigmoid_derivative_function(z_value)
        error = cost_derived * activation_derived
        errors.append(error)
    return errors


def regular_layer_error_calculator(layer_z_values, next_layer_weights, next_layer_errors):
    errors = []

    transposed_weights = np.array(np.transpose(np.array(next_layer_weights)))
    next_layer_errors_with_weights_applied = transposed_weights.dot(np.array(next_layer_errors))
    for i, val in enumerate(layer_z_values):
        vectorized_sigmoid_derivative_function = np.vectorize(sigmoid_derivative_function)
        derived_sigmoid = vectorized_sigmoid_derivative_function(val)
        error = derived_sigmoid * next_layer_errors_with_weights_applied[i]
        errors.append(error)

    return errors

def vectorized_result(labels, num_outputs):
    a = np.array(labels)
    vectorized = np.zeros((len(labels),num_outputs))
    vectorized[np.arange(len(labels)), a] = 1

    return np.transpose(vectorized)


def load_mnist_images(data, set, size=None, shuffled=False):
    if size is None:
        size = len(data[set]._images)
    inputs = list(data[set]._images[0:size])
    desired = list(data[set]._labels[0:size])
    if shuffled == True:
        combined = list(zip(inputs, desired))
        shuffle(combined)
        inputs[:], desired[:] = zip(*combined)
    return size, inputs, desired


def run_mini_batch(input_batch, desired_batch, neural_network, network_tuner):
    batch_correct = 0
    transposed_input_batch = np.transpose(input_batch)
    neural_network.batch_evaluate(transposed_input_batch)
    current_batch_desired_outputs = vectorized_result(desired_batch, len(neural_network.get_output_node_activations()))
    network_tuner.post_process(current_batch_desired_outputs, transposed_input_batch)
    guessed = neural_network.get_batched_highest_output()
    for i in range(len(desired_batch)):
        if desired_batch[i] == guessed[i]:
            batch_correct += 1
    # for i in range(0, len(desired_batch)):
    #     neural_network.evaluate(input_batch[i])
    #     current_desired = vectorized_result(desired_batch[i], len(neural_network.get_output_node_activations()))
    #     network_tuner.post_process(current_desired, input_batch[i])
    #     guessed = neural_network.get_highest_output()
    #     if desired_batch[i] == guessed:
    #         batch_correct += 1
    #     batch_total += 1
    #     network_tuner.calculate_cost(current_desired)
    return batch_correct, len(desired_batch)
