from Utils import Utils
from Node import Node


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def evaluate(self):
        result = []
        for i, node in enumerate(self.layers[-1].nodes):
            result.append(node.get_activation())

        return result

    @staticmethod
    def build(network_config, layer_builder):
        layers = []
        previous_layer = None
        for layer_index in range(network_config.number_layers):
            layer = layer_builder(previous_layer, str(layer_index), network_config.nodes_per_layer[layer_index],
                                  network_config.node_weight_provider, network_config.node_bias_provider, network_config.sigmoid_function, Node.build)
            layers.append(layer)
            previous_layer = layer
        return NeuralNetwork(layers)

    def set_input(self, inputs):
        for i, node in enumerate(self.layers[0].nodes):
            node.set_activation(inputs[i])
