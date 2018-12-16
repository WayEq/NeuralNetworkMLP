from NeuralNetworkMLP import NeuralNetwork
from NetworkPerformanceTuner import NetworkPerformanceTuner
from NetworkConfig import NetworkConfig
from NetworkLayer import NetworkLayer
from Utils import Utils
import pickle
import numpy as np
np.set_printoptions(precision=4)
from tensorflow.examples.tutorials.mnist import input_data


def output_layer_error_calculator(node_activations, node_z_values, desired_output):
    errors = []
    for i, z_value in enumerate(node_z_values):
        cost_derived = Utils.cost_derivative_function(desired_output[i], node_activations[i])
        activation_derived = Utils.sigmoid_derivative_function(z_value)
        error = cost_derived * activation_derived
        errors.append(error)
    return errors


def regular_layer_error_calculator(layer_z_values, next_layer_weights, next_layer_errors):
    errors = []

    transposed_weights = np.array(np.transpose(np.array(next_layer_weights)))
    next_layer_errors_with_weights_applied = transposed_weights.dot(np.array(next_layer_errors))
    for i, val in enumerate(layer_z_values):
        derived_sigmoid = Utils.sigmoid_derivative_function(val)
        error = derived_sigmoid * next_layer_errors_with_weights_applied[i]
        errors.append(error)

    return errors


def run_mini_batch(input_batch, desired_batch):
    batch_correct = 0
    batch_total = 0
    for i in range(0,len(desired_batch)):
        neural_network.evaluate(input_batch[i])
        network_tuner.increment_evaluations()
        current_desired = vectorized_result(desired_batch[i])
        errors = network_tuner.calculate_errors(current_desired)
        network_tuner.calculate_cost_gradient(errors, input_batch[i])
        guessed = neural_network.get_highest_output()
        if desired_batch[i] == guessed:
            batch_correct += 1
        batch_total += 1
        network_tuner.calculate_cost(current_desired)
    return batch_correct, batch_total


number_layers = 4
nodes_per_layer = [784, 16, 16, 10]

config = NetworkConfig(number_layers, nodes_per_layer, Utils.node_weight_provider, Utils.node_bias_provider,
                       Utils.sigmoid_function, Utils.sigmoid_derivative_function)

neural_network = NeuralNetwork.build(config, NetworkLayer.build)

load = False
if load:
    binary_file = open('network.bin',mode='rb')
    network = pickle.load(binary_file)
    neural_network.load(network.get_layers())

learning_rate = 1
network_tuner = NetworkPerformanceTuner(neural_network, regular_layer_error_calculator, output_layer_error_calculator,
                                        Utils.cost_function, learning_rate, config)


mini_batch_size = 10
epochs = 5


def vectorized_result(label):
    num_outputs = len(neural_network.get_output_node_activations())
    vectorized = [0 for _ in range(num_outputs)]
    vectorized[label] = 1
    return vectorized


training_set_size = 60000
data = input_data.read_data_sets("data/")
inputs = data[0]._images[0:training_set_size]
desired = data[0]._labels[0:training_set_size]

total_correct = 0
total_total = 0
for e in range(0,epochs):
    mini_batch_index = 0
    epoch_correct = 0
    epoch_total = 0
    while mini_batch_index < len(inputs):
        terminal = mini_batch_index+mini_batch_size
        input_batch = inputs[mini_batch_index:terminal]
        desired_batch = desired[mini_batch_index:terminal]
        (correct, total) = run_mini_batch(input_batch, desired_batch)
        epoch_correct += correct
        epoch_total += total
        network_tuner.tune()
        network_tuner.end_of_batch_reset()
        mini_batch_index = terminal
    print("Epoch totals: " + str(epoch_correct) + " / " + str(epoch_total) + " (" + str(epoch_correct/epoch_total) + ")")
    total_correct += epoch_correct
    total_total += epoch_total

print("Totals: " + str(total_correct) + " / " + str(total_total) + " (" + str(total_correct / total_total) + ")")

binary_file = open('network.bin',mode='wb')
my_pickled_mary = pickle.dump(neural_network, binary_file)
binary_file.close()

