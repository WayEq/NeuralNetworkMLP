import numpy as np
np.set_printoptions(precision=4)


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def evaluate(self):
        list(map(lambda my_layer: my_layer.clear_activations(), self.layers[1:]))
        previous_layer_activations = self.layers[0].get_activations()
        for i, layer in enumerate(self.layers[1:]):
            layer.feed_forward(previous_layer_activations)
            previous_layer_activations = layer.get_activations()

    @staticmethod
    def build(network_config, layer_builder):
        layers = []
        number_of_upstream_nodes = 0
        for layer_index in range(network_config.number_layers):
            layer = layer_builder(number_of_upstream_nodes, str(layer_index),
                                  network_config.nodes_per_layer[layer_index],
                                  network_config.node_weight_provider, network_config.node_bias_provider,
                                  network_config.sigmoid_function)
            layers.append(layer)
            number_of_upstream_nodes = layer.get_number_of_nodes()
        return NeuralNetwork(layers)

    def load(self, layers):
        self.layers = layers

    def get_layers(self):
        return self.layers

    def get_output_node_activations(self):
        return self.layers[-1].get_activations()

    def get_highest_output(self):
        activations = self.get_output_node_activations()
        return activations.index(max(activations))

    def set_input(self, inputs):
        self.layers[0].set_activations(inputs)
