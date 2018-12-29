import numpy as np

# Responsible for evaluating network performance and telling the network what parameters to change

debug = False


def calculate_weight_gradients(node_error, upstream_activations):
    return np.multiply(upstream_activations, node_error)


class NetworkPerformanceTuner:
    def __init__(self, network, hidden_layer_error_calculator, output_layer_error_calculator, cost_calculator,
                 learning_rate, network_config):
        self.network_config = network_config
        self.network = network
        self.layer_error_calculator = hidden_layer_error_calculator
        self.output_layer_error_calculator = output_layer_error_calculator
        self.learning_rate = learning_rate
        self.cost_calculator = cost_calculator

        # Stuff that gets reset after a tuning
        self.weight_gradient = []
        self.bias_gradient = []
        self.average_cost = 0
        self.previous_batch_cost = 0

    def calculate_cost(self, desired_output):
        output_activations = self.network.get_output_node_activations()
        cost = self.cost_calculator(desired_output, output_activations)
        return cost

    def tune(self):
        (weight_variable_deltas, bias_variable_deltas) = self.__calculate_variable_deltas()

        self.__apply_deltas(weight_variable_deltas, bias_variable_deltas)
        self.__end_of_batch_reset()

    def post_process(self, desired, input_layer_activations):
        errors = self.__calculate_errors(desired)
        self.__calculate_cost_gradient(errors, input_layer_activations)

    def __end_of_batch_reset(self):
        self.weight_gradient = []
        self.bias_gradient = []
        self.previous_batch_cost = self.average_cost
        self.average_cost = 0

    def __calculate_layer_errors(self, desired_output):
        errors = []
        layers = self.network.layers
        output_layer = layers[-1]
        errors.append(self.output_layer_error_calculator(output_layer.activations, output_layer.z_values,
                                                         desired_output))
        next_layer = output_layer
        next_layer_errors = errors[0]
        for layer in reversed((layers[:-1])):
            this_layer_z_values = layer.z_values
            next_layer_weights = next_layer.weights
            this_layer_errors = self.layer_error_calculator(this_layer_z_values, next_layer_weights, next_layer_errors)
            errors.insert(0, this_layer_errors)
            next_layer = layer
            next_layer_errors = this_layer_errors
        if debug:
            print("Calculated errors: " + str(errors))
        return errors

    def __calculate_errors(self, desired):
        errors = self.__calculate_layer_errors(desired)
        return errors

    def __calculate_cost_gradient(self, errors, input_layer_activations):
        layers = self.network.layers
        previous_layer_activations = input_layer_activations
        for layer_index, layer in enumerate(layers):
            self.__calculate_layer_cost_gradient(errors[layer_index], layer.number_of_nodes,
                                                 previous_layer_activations, layer_index)
            previous_layer_activations = layer.activations
        if debug:
            print("Calculated weight gradient: " + str(self.weight_gradient))
            print("Calculated bias gradient: " + str(self.bias_gradient))

    def __calculate_layer_cost_gradient(self, node_errors, number_of_nodes, previous_layer_activations, layer_index):
        needs_initialization = len(self.weight_gradient) == layer_index
        if needs_initialization:
            this_array_gradient = np.zeros((number_of_nodes, len(previous_layer_activations), len(previous_layer_activations[0])))
            self.weight_gradient.append(this_array_gradient)
            this_bias_gradient = np.zeros((number_of_nodes, len(previous_layer_activations[0])))
            self.bias_gradient.append(this_bias_gradient)

        # Add to the weights cost gradients
        for node_index, node_error in enumerate(node_errors):
            weight_gradients = calculate_weight_gradients(node_error, previous_layer_activations)
            self.weight_gradient[layer_index][node_index, :] += weight_gradients

        # Add to the biases cost gradients
        self.bias_gradient[layer_index] += node_errors

    def __calculate_variable_deltas(self):
        weight_variable_deltas = []
        bias_variable_deltas = []
        batch_size = len(self.weight_gradient[0][0][0])
        for layer_index, layer in enumerate(self.network.layers):
            layer_weight_deltas = self.weight_gradient[layer_index] * self.learning_rate * -1 / batch_size
            weight_variable_deltas.append(layer_weight_deltas)
            layer_bias_deltas = self.bias_gradient[
                                    layer_index] * self.learning_rate * -1 / batch_size
            bias_variable_deltas.append(layer_bias_deltas)
        if debug:
            print("Calculated variable deltas: " + str(weight_variable_deltas) + " bias: " + str(bias_variable_deltas))
        return weight_variable_deltas, bias_variable_deltas

    def __apply_deltas(self, weight_variable_deltas, bias_variable_deltas):
        for i, layer in enumerate(self.network.layers):
            layer.apply_deltas(weight_variable_deltas[i], bias_variable_deltas[i])
