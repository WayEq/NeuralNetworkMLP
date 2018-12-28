from NetworkPerformanceTuner import NetworkPerformanceTuner
from NeuralNetworkMLP import NeuralNetwork
from NetworkConfig import NetworkConfig
from NetworkLayer import NetworkLayer
import Utils
import pickle
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

np.set_printoptions(precision=4)

nodes_per_layer = [784, 16, 16, 10]

config = NetworkConfig(nodes_per_layer, Utils.node_weight_provider, Utils.node_bias_provider,
                       Utils.sigmoid_function, Utils.sigmoid_derivative_function)

neural_network = NeuralNetwork.build(config, NetworkLayer.build)

load = False
if load:
    binary_file = open('network.bin',mode='rb')
    network = pickle.load(binary_file)
    neural_network.load(network.get_layers())

learning_rate = 1
network_tuner = NetworkPerformanceTuner(neural_network, Utils.regular_layer_error_calculator,
                                        Utils.cross_entropy_output_layer_calculator,
                                        Utils.cross_entropy_cost_function, learning_rate, config)


mini_batch_size = 20
epochs = 10

training_set_size = 60000
data = input_data.read_data_sets("data/")


total_correct = 0
total_total = 0

for e in range(0, epochs):
    inputs, desired = Utils.shuffle_training_data(data, training_set_size)
    mini_batch_index = 0
    epoch_correct = 0
    epoch_total = 0
    while mini_batch_index < len(inputs):
        terminal = mini_batch_index+mini_batch_size
        input_batch = inputs[mini_batch_index:terminal]
        desired_batch = desired[mini_batch_index:terminal]
        (correct, total) = Utils.run_mini_batch(input_batch, desired_batch, neural_network, network_tuner)
        epoch_correct += correct
        epoch_total += total
        network_tuner.tune()
        mini_batch_index = terminal
    print("Epoch totals: " + str(epoch_correct) + " / "
          + str(epoch_total) + " (" + str(epoch_correct/epoch_total) + ")")
    total_correct += epoch_correct
    total_total += epoch_total

print("Totals: " + str(total_correct) + " / " + str(total_total) + " (" + str(total_correct / total_total) + ")")

binary_file = open('network.bin', mode='wb')
pickle.dump(neural_network, binary_file)
binary_file.close()
