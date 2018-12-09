from Node import Node


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def evaluate(self):
        return self.layers[-1].get_activation()

    @staticmethod
    def build(network_config, layer_builder):
        layers = []
        previous_layer = None
        for layer_index in range(network_config.number_layers):
            layer = layer_builder(previous_layer, str(layer_index), network_config.nodes_per_layer[layer_index],
                                  network_config.node_weight_provider, network_config.node_bias_provider,
                                  network_config.sigmoid_function, network_config.sigmoid_derivative_function,
                                  Node.build)
            layers.append(layer)
            previous_layer = layer
        return NeuralNetwork(layers)

    def get_layers(self):
        return self.layers

    def set_input(self, inputs):
        for i, node in enumerate(self.layers[0].nodes):
            node.set_activation(inputs[i])


# Responsible for evaluating network performance and telling the network what parameters to change
class NetworkPerformanceTuner:
    def __init__(self, network, hidden_layer_error_calculator, output_layer_error_calculator):
        self.network = network
        self.layer_error_calculator = hidden_layer_error_calculator
        self.output_layer_error_calculator = output_layer_error_calculator
        self.errors = None

    def calculate_layer_errors(self, layers, desired_output):
        this_layer = layers[0]
        if len(layers) == 1:
            # This is the output layer and the end of our recursive case
            return [self.output_layer_error_calculator(this_layer, desired_output)]

        this_layer_z_values = this_layer.get_z_values()
        next_layer_weights = layers[1].get_weights()
        next_layer_errors = self.calculate_layer_errors(layers[1:], desired_output)
        this_layer_errors = self.layer_error_calculator(this_layer_z_values, next_layer_weights, next_layer_errors)
        next_layer_errors.insert(0, this_layer_errors)
        return next_layer_errors

    def calculate_errors(self, desired):
        self.errors = self.calculate_layer_errors(self.network.get_layers(), desired)
        print("Calculated errors: " + str(self.errors))
