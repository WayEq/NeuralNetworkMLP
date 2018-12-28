import numpy as np
np.set_printoptions(precision=4)

debug = False

class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def evaluate(self, input_activations):
        previous_layer_activations = input_activations
        for layer in self.layers:
            layer.feed_forward(previous_layer_activations)
            previous_layer_activations = layer.get_activations()
        if debug:
            print("Evaluated to: " + str(self.get_output_node_activations()))

    def batch_evaluate(self, input_activations):
        previous_layer_activations = input_activations
        for layer in self.layers:
            layer.batch_feed_forward(np.array(previous_layer_activations))
            previous_layer_activations = layer.get_activations()

    @staticmethod
    def build(network_config, layer_builder):
        layers = []
        number_of_upstream_nodes = network_config.nodes_per_layer[0]
        for layer_index in range(len(network_config.nodes_per_layer) -1):
            layer = layer_builder(number_of_upstream_nodes, str(layer_index),
                                  network_config.nodes_per_layer[layer_index+1],
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

    def get_batched_highest_output(self):
        activations = self.get_output_node_activations()
        return activations.argmax(axis=0)

    def display(self):
        for layer in self.layers:
            layer.display()
