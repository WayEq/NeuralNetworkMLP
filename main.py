from NeuralNetworkMLP import NeuralNetwork, NetworkPerformanceTuner
from NetworkConfig import NetworkConfig
from NetworkLayer import NetworkLayer
from Utils import Utils

config = NetworkConfig(3, [1, 2, 1], Utils.node_weight_provider, Utils.node_bias_provider, Utils.sigmoid_function,
                       Utils.sigmoid_derivative_function)
desired = [1]
neural_network = NeuralNetwork.build(config, NetworkLayer.build)
inputs = [1]
neural_network.set_input(inputs)
result = neural_network.evaluate()


def output_layer_error_calculator(layer, desired_output):
    errors = []
    for i, node in enumerate(layer.get_nodes()):
        cost_derived = Utils.cost_derivative_function(desired_output[i], node.get_activation())
        print("Cost derived: " + str(cost_derived))
        activation_derived = Utils.sigmoid_derivative_function(node.get_z_value())
        error = cost_derived * activation_derived
        errors.append(error)
    return errors


def regular_layer_error_calculator(layer, next_layer_weights, next_layer_errors):
    errors = []
    for i, val in enumerate(layer):
        # TODO calculate errors
        errors.append(i)
    return errors


network_tuner = NetworkPerformanceTuner(neural_network, regular_layer_error_calculator, output_layer_error_calculator)
network_tuner.calculate_errors(desired)
