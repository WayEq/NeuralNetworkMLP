from NeuralNetworkMLP import NeuralNetwork
from NetworkPerformanceTuner import NetworkPerformanceTuner
from NetworkConfig import NetworkConfig
from NetworkLayer import NetworkLayer
from Utils import Utils
import numpy as np
np.set_printoptions(precision=4)



def output_layer_error_calculator(node_activations, node_z_values, desired_output):
    errors = []
    for i, z_value in enumerate(node_z_values):
        cost_derived = Utils.cost_derivative_function(desired_output[i], node_activations[i])
        #print("Cost derived: " + str(cost_derived))
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
    # print("errors: " + str(errors))

    return errors


def run_mini_batch():
    correct = 0
    total = 0
    for i in range(0,len(desired)):
        neural_network.set_input(inputs[i])
        neural_network.evaluate()
        network_tuner.incr_evaluations()
        current_desired = desired[i]
        errors = network_tuner.calculate_errors(current_desired)
        network_tuner.calculate_cost_gradient(errors)
        #print("For desired: " + str(current_desired) + " i guessed: " + str(neural_network.get_output_node_activations()))
        desired_number = i
        guessed = neural_network.get_highest_output()
        # print("Desired: " + str(desired_number) + " guessed: " + str(guessed))
        if desired_number == guessed:
            correct += 1
        total += 1
        network_tuner.calculate_cost(current_desired)
    print("Batch correct %: " + str(correct / total))

config = NetworkConfig(4, [10, 16, 16, 10], Utils.node_weight_provider, Utils.node_bias_provider, Utils.sigmoid_function,
                       Utils.sigmoid_derivative_function)
neural_network = NeuralNetwork.build(config, NetworkLayer.build)

learning_rate = .2
network_tuner = NetworkPerformanceTuner(neural_network, regular_layer_error_calculator, output_layer_error_calculator,
                                        Utils.cost_function, learning_rate)

def get(j):
    l = [ 0 for j in range(0,10)]
    l[j] = 1
    return l


inputs = [get(j) for j in [ i for i in range(0,10)]]
desired = inputs


iterations = 5000

for i in range(0,iterations):
    run_mini_batch()
    network_tuner.tune()
    network_tuner.reset()
