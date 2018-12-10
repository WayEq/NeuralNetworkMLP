from NeuralNetworkMLP import NeuralNetwork, NetworkPerformanceTuner
from NetworkConfig import NetworkConfig
from NetworkLayer import NetworkLayer
from Utils import Utils
import numpy as np
np.set_printoptions(precision=4)



def output_layer_error_calculator(node_activations, node_z_values, desired_output):
    errors = []
    for i, z_value in enumerate(node_z_values):
        cost_derived = Utils.cost_derivative_function(desired_output[i], node_activations[i])
        print("Cost derived: " + str(cost_derived))
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
    print("errors: " + str(errors))

    return errors


config = NetworkConfig(3, [1, 2, 1], Utils.node_weight_provider, Utils.node_bias_provider, Utils.sigmoid_function,
                       Utils.sigmoid_derivative_function)
neural_network = NeuralNetwork.build(config, NetworkLayer.build)
network_tuner = NetworkPerformanceTuner(neural_network, regular_layer_error_calculator, output_layer_error_calculator,
                                        Utils.cost_function)
desired = [1]
inputs = [1]
neural_network.set_input(inputs)

for i in range(0,100):
    result = neural_network.evaluate()
    network_tuner.calculate_errors(desired)
    network_tuner.calculate_cost_gradient()
    network_tuner.calculate_variable_deltas()
    network_tuner.calculate_cost(desired)
    network_tuner.apply_deltas()
    print()

