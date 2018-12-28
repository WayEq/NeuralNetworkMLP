import NetworkLayer
import Utils
from NetworkConfig import NetworkConfig
from NetworkPerformanceTuner import NetworkPerformanceTuner
from NeuralNetworkMLP import NeuralNetwork
import numpy as np

nodes_per_layer = [3,2]

weights = 0

def weight_provider(num):
    return np.arange(num)

config = NetworkConfig(nodes_per_layer, weight_provider, Utils.node_bias_provider,
                       Utils.sigmoid_function, Utils.sigmoid_derivative_function)

neural_network = NeuralNetwork.build(config, NetworkLayer.NetworkLayer.build)


learning_rate = 1
network_tuner = NetworkPerformanceTuner(neural_network, Utils.regular_layer_error_calculator,
                                        Utils.cross_entropy_output_layer_calculator,
                                        Utils.cross_entropy_cost_function, learning_rate, config)


inputs = np.arange(6.0).reshape(2,3)
desired_outputs = np.array((1, 0))
while True:
    batch_correct, total = Utils.run_mini_batch(inputs, desired_outputs, neural_network, network_tuner)
    network_tuner.tune()
    print(str(batch_correct / total))


# neural_network.batch_evaluate(np.arange(6.0).reshape(3,2))
# activations = neural_network.get_output_node_activations()
#
# print("activations: " + str(activations))

