import numpy as np

# Responsible for evaluating network performance and telling the network what parameters to change

debug = False


class NetworkPerformanceTuner:
    def __init__(self, network, hidden_layer_error_calculator, output_layer_error_calculator, cost_calculator,
                 learning_rate, network_config, training_data_size, regularization_term=0):
        self.training_data_size = training_data_size
        self.network_config = network_config
        self.network = network
        self.layer_error_calculator = hidden_layer_error_calculator
        self.output_layer_error_calculator = output_layer_error_calculator
        self.learning_rate = learning_rate
        self.cost_calculator = cost_calculator

        # Stuff that gets reset after a tuning
        self.weight_gradient = []
        self.bias_gradient = []
        self.regularization_term = regularization_term

    def calculate_batch_cost(self, desired_output):
        output_activations = self.network.get_output_node_activations()
        weights = self.network.get_weights()
        return self.cost_calculator(desired_output, output_activations, weights, self.training_data_size, self.regularization_term)

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
        self.weight_gradient = []
        self.bias_gradient = []
        previous = [layer.activations for layer in self.network.layers[:-1]]
        previous.insert(0,input_layer_activations)
        for layer_index, layer in enumerate(layers):
            self.__calculate_layer_cost_gradient(errors[layer_index], previous_layer_activations, layer.weights)
            previous_layer_activations = layer.activations
        if debug:
            print("Calculated weight gradient: " + str(self.weight_gradient))
            print("Calculated bias gradient: " + str(self.bias_gradient))

    def __calculate_layer_cost_gradient(self, node_errors, previous_layer_activations, weights):

        # Add to the weights cost gradients
        weight_gradients = np.average(
            np.matmul(np.transpose(node_errors)[:, :, np.newaxis],
                      np.transpose(previous_layer_activations)[:, np.newaxis, :]), axis=0)
        # Add the regularization costs (if any)
        weight_gradients += (self.regularization_term / self.training_data_size) * np.array(weights)
        self.weight_gradient.append(weight_gradients)

        # Add to the biases cost gradients
        self.bias_gradient.append(np.average(node_errors,axis=1))

    def __calculate_variable_deltas(self):
        weight_variable_deltas = np.array(self.weight_gradient) * self.learning_rate * -1
        bias_variable_deltas = np.array(self.bias_gradient) * self.learning_rate * -1
        return weight_variable_deltas, bias_variable_deltas

    def __apply_deltas(self, weight_variable_deltas, bias_variable_deltas):
        for i, layer in enumerate(self.network.layers):
            layer.apply_deltas(weight_variable_deltas[i], bias_variable_deltas[i])
