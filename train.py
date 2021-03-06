from NetworkPerformanceTuner import NetworkPerformanceTuner
from NeuralNetworkMLP import NeuralNetwork
from NetworkConfig import NetworkConfig
from NetworkLayer import NetworkLayer
import Utils
import pickle
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

np.set_printoptions(precision=4)


def execute_verification(data, epoch, neural_network):
    size, inputs, desired = Utils.load_mnist_images(data, 1)
    verification_correct = 0
    verification_total = 0
    for e in range(0, size):
        neural_network.evaluate(inputs[e])
        output = neural_network.get_highest_output()
        if output == desired[e]:
            verification_correct += 1
        verification_total += 1

    print("Verification accuracy (" + str(epoch) + "): "
          + str(verification_correct) + " / " + str(verification_total) + " (" +
          str(verification_correct / verification_total) + ")")


def train():
    nodes_per_layer = [784, 100, 10]
    mini_batch_size = 10
    epochs = 60
    learning_rate = .1
    load = False
    regularization_term = 5
    training_set_size = 1000
    config = NetworkConfig(nodes_per_layer, Utils.node_weight_provider, Utils.node_bias_provider,
                           Utils.sigmoid_function, Utils.sigmoid_derivative_function)

    neural_network = NeuralNetwork.build(config, NetworkLayer.build)

    network_tuner = NetworkPerformanceTuner(neural_network, Utils.regular_layer_error_calculator,
                                            Utils.cross_entropy_output_layer_calculator,
                                            Utils.batch_cross_entropy_cost_function_with_l2_regularize, learning_rate, config, training_set_size, regularization_term)


    if load:
        binary_file = open('network.bin', mode='rb')
        network = pickle.load(binary_file)
        neural_network.load(network.layers)

    data = input_data.read_data_sets("data/")
    total_correct = 0
    total_total = 0
    for e in range(0, epochs):
        size, inputs, desired = Utils.load_mnist_images(data, 0, training_set_size, shuffled=True)
        mini_batch_index = 0
        epoch_correct = epoch_total = 0
        epoch_cost = 0
        print_batch_totals = False
        while mini_batch_index < len(inputs):
            terminal = mini_batch_index+mini_batch_size
            input_batch = inputs[mini_batch_index:terminal]
            desired_batch = desired[mini_batch_index:terminal]
            (correct, total) = Utils.run_mini_batch(input_batch, desired_batch, neural_network, network_tuner)
            epoch_correct += correct
            epoch_total += total
            batch_cost = network_tuner.calculate_batch_cost(desired_batch)
            epoch_cost += batch_cost
            network_tuner.tune()
            mini_batch_index = terminal
            if print_batch_totals:
                print("Epoch totals: " + str(epoch_correct) + " / "
                      + str(epoch_total) + " (" + str(epoch_correct/epoch_total) + ")")
        total_correct += epoch_correct
        total_total += epoch_total
        #print("Epoch ("+ str(e) + ") average training cost: " + str(epoch_cost / (len(inputs) / mini_batch_size)))

        execute_verification(data, e, neural_network)

    print("Totals: " + str(total_correct) + " / " + str(total_total) + " (" + str(total_correct / total_total) + ")")
    binary_file = open('network.bin', mode='wb')
    pickle.dump(neural_network, binary_file)
    binary_file.close()


train()
