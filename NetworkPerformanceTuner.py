from Utils import Utils
import numpy as np
import random

# Responsible for evaluating network performance and telling the network what parameters to change

debug = False


def calculate_weight_gradients(node_error, upstream_activations):
    return np.multiply(upstream_activations, node_error)


class NetworkPerformanceTuner:
    def __init__(self, network, hidden_layer_error_calculator, output_layer_error_calculator, cost_calculator,
                 learning_rate):
        self.network = network
        self.layer_error_calculator = hidden_layer_error_calculator
        self.output_layer_error_calculator = output_layer_error_calculator
        self.learning_rate = learning_rate
        self.cost_calculator = cost_calculator

        self.batch_evaluations_counter = 0
        self.weight_gradient = [np.array([])]  # The input layer doesn't have gradient or bias
        self.bias_gradient = [np.array([])]
        self.average_cost = 0
        self.previous_batch_cost = 0

    def end_of_batch_reset(self):
        self.batch_evaluations_counter = 0
        self.weight_gradient = [None]
        self.bias_gradient = [None]
        self.previous_batch_cost = self.average_cost
        self.average_cost = 0

    def calculate_cost(self, desired_output):
        output_activations = self.network.get_output_node_activations()
        cost = self.cost_calculator(output_activations, desired_output)
        self.average_cost = Utils.add_to_average(self.average_cost, self.batch_evaluations_counter, cost)

    def calculate_layer_errors(self, desired_output):
        errors = []
        layers = self.network.get_layers()
        output_layer = layers[-1]
        errors.append(self.output_layer_error_calculator(output_layer.get_activations(), output_layer.get_z_values(),
                                                         desired_output))
        next_layer = output_layer
        next_layer_errors = errors[0]
        for layer in reversed((layers[1:-1])):
            this_layer_z_values = layer.get_z_values()
            next_layer_weights = next_layer.get_weights()
            this_layer_errors = self.layer_error_calculator(this_layer_z_values, next_layer_weights, next_layer_errors)
            errors.insert(0, this_layer_errors)
            next_layer = layer
            next_layer_errors = this_layer_errors
        errors.insert(0, None)
        if debug:
            print("Calculated errors: " + str(errors))
        return errors

    def increment_evaluations(self):
        self.batch_evaluations_counter += 1

    def calculate_errors(self, desired):
        errors = self.calculate_layer_errors(desired)
        return errors

    def calculate_cost_gradient(self, errors):
        layers = self.network.get_layers()

        for layer_index, layer in enumerate(layers[1:], 1):
            previous_layer = layers[layer_index - 1]
            self.calculate_layer_cost_gradient(errors[layer_index], layer, previous_layer, layer_index)
        if debug:
            print("Calculated weight gradient: " + str(self.weight_gradient))
        if debug:
            print("Calculated bias gradient: " + str(self.bias_gradient))

    def calculate_layer_cost_gradient(self, node_errors, layer, previous_layer, layer_index):
        needs_initialization = len(self.weight_gradient) == layer_index
        if needs_initialization:
            number_of_nodes = layer.get_number_of_nodes()
            this_array_gradient = np.zeros((number_of_nodes, previous_layer.get_number_of_nodes()))
            self.weight_gradient.append(this_array_gradient)
            self.bias_gradient.append(np.zeros(number_of_nodes))
        upstream_activations = np.array(previous_layer.get_activations())

        # Add to the weights cost gradients
        for node_index, node_error in enumerate(node_errors):
            self.weight_gradient[layer_index][node_index, :] \
                += calculate_weight_gradients(node_error, upstream_activations)

        # Add to the biases cost gradients
        self.bias_gradient[layer_index] += node_errors

    def calculate_variable_deltas(self):
        weight_variable_deltas = [None]
        bias_variable_deltas = [None]

        for layer_index, layer in enumerate(self.network.get_layers()[1:], 1):
            previous_layer = self.network.get_layers()[layer_index - 1]
            needs_initialization = len(weight_variable_deltas) == layer_index
            if needs_initialization:
                number_of_nodes = layer.get_number_of_nodes()
                weight_variable_deltas.append(np.array((number_of_nodes, previous_layer.get_number_of_nodes())))
                bias_variable_deltas.append(np.array(number_of_nodes))
            weight_variable_deltas[layer_index] = self.weight_gradient[layer_index] \
                * self.learning_rate * -1 / self.batch_evaluations_counter

            bias_variable_deltas[layer_index] = self.bias_gradient[layer_index] \
                * self.learning_rate * -1 / self.batch_evaluations_counter
        if debug:
            print("Calculated variable deltas: " + str(weight_variable_deltas) + " bias: " + str(bias_variable_deltas))
        return weight_variable_deltas, bias_variable_deltas

    def apply_deltas(self, weight_variable_deltas, bias_variable_deltas):
        for i, layer in enumerate(self.network.get_layers()[1:], 1):
            layer.apply_deltas(weight_variable_deltas[i], bias_variable_deltas[i])

    def tune(self):
        cost_improvement = self.average_cost - self.previous_batch_cost
        if random.random() > .99:
            print("Average cost over this batch: " + str(self.average_cost) + " (" + str(cost_improvement) + ")")
        (weight_variable_deltas, bias_variable_deltas) = self.calculate_variable_deltas()
        self.apply_deltas(weight_variable_deltas, bias_variable_deltas)
