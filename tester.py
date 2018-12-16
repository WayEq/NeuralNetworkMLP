import pickle

import numpy as np

from NetworkConfig import NetworkConfig
from NetworkLayer import NetworkLayer
from NeuralNetworkMLP import NeuralNetwork
from Utils import Utils
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.set_printoptions(precision=4)


number_layers = 4
nodes_per_layer = [784, 16, 16, 10]
config = NetworkConfig(number_layers, nodes_per_layer, Utils.node_weight_provider, Utils.node_bias_provider,
                       Utils.sigmoid_function, Utils.sigmoid_derivative_function)

neural_network = NeuralNetwork.build(config, NetworkLayer.build)

load = True
if load:
    binary_file = open('network.bin', mode='rb')
    network = pickle.load(binary_file)
    neural_network.load(network.get_layers())

training_set_size = 10000
data = input_data.read_data_sets("data/")
inputs = data[2]._images[0:training_set_size]
desired = data[2]._labels[0:training_set_size]

total_correct = 0
total_total = 0
for e in range(0,training_set_size):
    neural_network.evaluate(inputs[e])
    output = neural_network.get_highest_output()
    print("I guessed " + str(output) + " actually " + str(desired[e]))
    if output == desired[e]:
        total_correct += 1
    total_total += 1
    img1_2d = np.reshape(inputs[e], (28, 28))
    # show it
    plt.subplot(111)
    plt.imshow(img1_2d, cmap=plt.get_cmap('gray'))
    plt.show()

print("Totals: " + str(total_correct) + " / " + str(total_total) + " (" + str(total_correct / total_total) + ")")
