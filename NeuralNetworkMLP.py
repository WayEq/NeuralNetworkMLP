import NetworkLayer
from Node import Node
import numpy as np
np.set_printoptions(precision=4)

class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def evaluate(self):
        list(map(lambda layer: layer.clear_activations(), self.layers[1:]))
        return self.layers[-1].get_activations()

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

    def get_output_node_activations(self):
        return self.layers[-1].get_activations()

    def set_input(self, inputs):
        for i, node in enumerate(self.layers[0].nodes):
            node.set_activation(inputs[i])


# Responsible for evaluating network performance and telling the network what parameters to change
class NetworkPerformanceTuner:
    def __init__(self, network, hidden_layer_error_calculator, output_layer_error_calculator, cost_calculator):
        self.network = network
        self.layer_error_calculator = hidden_layer_error_calculator
        self.output_layer_error_calculator = output_layer_error_calculator
        self.errors = None
        self.gradient = None
        self.variable_deltas = None
        self.cost_calculator = cost_calculator

    def calculate_cost(self, desired_output):
        output_activations = self.network.get_output_node_activations()
        cost = self.cost_calculator(output_activations, desired_output)
        print("Calculated cost: " + str(cost))

    def calculate_layer_errors(self, layers, desired_output):
        this_layer = layers[0]
        if len(layers) == 1:
            # This is the output layer and the end of our recursive case
            return [self.output_layer_error_calculator(this_layer.get_activations(), this_layer.get_z_values(), desired_output)]

        this_layer_z_values = this_layer.get_z_values()
        next_layer_weights = layers[1].get_weights()
        all_subsequent_layer_errors = self.calculate_layer_errors(layers[1:], desired_output)
        next_layer_errors = all_subsequent_layer_errors[0]
        this_layer_errors = self.layer_error_calculator(this_layer_z_values, next_layer_weights, next_layer_errors)
        all_subsequent_layer_errors.insert(0, this_layer_errors)
        return all_subsequent_layer_errors

    def calculate_errors(self, desired):
        self.errors = self.calculate_layer_errors(self.network.get_layers()[1:], desired)
        print("Calculated errors: " + str(self.errors))

    def calculate_cost_gradient(self):
        gradient = []

        for i, layer in enumerate(self.network.get_layers()[1:]):
            layer_gradients = []

            for j, node in enumerate(layer.get_nodes()):

                node_gradients = []
                error = self.errors[i][j]
                for k, link in enumerate(node.get_links()):

                    node_gradients.append(error * link.upstream_node.get_activation())
                node_gradients.append(error)
                layer_gradients.append(node_gradients)
            gradient.append(layer_gradients)
        self.gradient = gradient

        print("gradient: " + str(gradient))

    def calculate_variable_deltas(self):
        learning_rate = .01
        rate_ = lambda variable: variable * (learning_rate * -1)
        node_ = lambda node: list(map(rate_, node))
        layer_ = lambda layer: list(map(node_, layer))
        deltas = list(map(layer_, self.gradient))
        self.variable_deltas = deltas
        print("deltas: " + str(deltas))

    def apply_deltas(self):
        for i, layer in enumerate(self.network.get_layers()[1:]):
            #print("layer: " + str(i+1))
            for j, node in enumerate(layer.get_nodes()):
                #print("node: " + str(j) + " old bias: " + str(node.get_bias()))

                for k, link in enumerate(node.get_links()):
                    #print("link: " + str(k), " old weight:" + str(link.weight))
                    link_delta = self.variable_deltas[i][j][k]
                    link.weight += link_delta
                    #print("new weight: " + str(link.weight))
                node.set_bias(node.get_bias() + self.variable_deltas[i][j][-1])
                #print("new node bias: " + str(node.get_bias()))