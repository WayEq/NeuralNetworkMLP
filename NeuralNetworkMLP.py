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

    def get_highest_output(self):
        activations = self.get_output_node_activations()
        return activations.index(max(activations))

    def set_input(self, inputs):
        for i, node in enumerate(self.layers[0].nodes):
            node.set_activation(inputs[i])
