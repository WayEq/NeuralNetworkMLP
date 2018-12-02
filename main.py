from NeuralNetworkMLP import NeuralNetwork
from NetworkConfig import NetworkConfig
from NetworkLayer import NetworkLayer
from Utils import Utils

config = NetworkConfig(3, [1, 2, 1], Utils.node_weight_provider, Utils.node_bias_provider, Utils.sigmoid_function)
desired = [1]
neural_network = NeuralNetwork.build(config, NetworkLayer.build)
inputs = [1]
neural_network.set_input(inputs)
result = neural_network.evaluate()
print("Result: " + str(result))
print("Cost: " + str(Utils.calculate_cost(desired, result)))
