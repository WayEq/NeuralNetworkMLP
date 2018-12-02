from functools import reduce
from scipy.stats import logistic
from random import uniform
from dataclasses import dataclass


class Link:
    def __init__(self, upstream_node, weight):
        self.upstream_node = upstream_node
        self.weight = weight


class Node:
    def __init__(self, node_id, links, bias):
        self.node_id = node_id
        self.activation = None
        self.links = links
        self.bias = bias

        self.node_print("initializing node")

    def node_print(self, message):
        print (self.node_id + " " + message)

    def get_activation(self):
        if self.activation is None:
            reduced = reduce(lambda a, link: a + self.compute_link_value(link), self.links, 0)
            to_squish = reduced + self.bias
            self.activation = round(logistic.cdf(to_squish),2)
            self.node_print("calculated my activation " +  str(self.activation))

        return self.activation

    def compute_link_value(self, link):
        self.node_print("Link activation " + str(link.upstream_node.get_activation()))
        self.node_print("Link weight " + str(link.weight))
        link_value = link.weight * link.upstream_node.get_activation()
        self.node_print("Link value " + str(link_value))
        return link_value

    def set_activation(self, activation):
        self.node_print("Setting activation of input node")
        self.activation = activation

    @staticmethod
    def build(node_id, upstream_layer, weights, bias):

        links = []
        if upstream_layer is None:
            return Node(node_id, [], bias)
        for i, upstream_node in enumerate(upstream_layer.nodes):
            links.append(Link(upstream_node, weights[i]))

        return Node(node_id, links, bias)


class NetworkLayer:

    def __init__(self, id, nodes):
        self.id = id
        self.nodes = nodes

    @staticmethod
    def build(upstream_layer, layer_id, number_nodes, my_node_weight_provider, my_node_bias_provider, node_builder):
        nodes = []
        upstream_size = 0 if upstream_layer is None else len(upstream_layer.nodes)
        for node_index in range(number_nodes):
            node_id = layer_id + str(node_index)
            nodes.append(node_builder(node_id, upstream_layer,
                                      my_node_weight_provider(upstream_size), my_node_bias_provider()))
        return NetworkLayer(layer_id, nodes)


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def evaluate(self):

        for i, node in enumerate(self.layers[-1].nodes):
            activation = node.get_activation()
            print("Activation for node " + str(i) + ": " + str(activation))

    @staticmethod
    def build(network_config, layer_builder):
        layers = []
        previous_layer = None
        for layer_index in range(network_config.number_layers):
            layer = layer_builder(previous_layer, str(layer_index), network_config.nodes_per_layer[layer_index], node_weight_provider,
                                  node_bias_provider, Node.build)
            layers.append(layer)
            previous_layer = layer
        return NeuralNetwork(layers)

    def set_input(self, inputs):
        for i, node in enumerate(self.layers[0].nodes):
            node.set_activation(inputs[i])


@dataclass
class NetworkConfig:
    number_layers: int
    nodes_per_layer: []
    node_bias_provider: lambda _: int
    node_weight_provider: lambda number_of_weights: []


def node_weight_provider(number_of_weights):
    weights = []
    for i in range(number_of_weights):
        weights.append(round(uniform(0, 1), 2))
    return weights


def node_bias_provider(): return 0


config = NetworkConfig(3, [1, 1, 1], node_weight_provider, node_bias_provider)
neural_network = NeuralNetwork.build(config, NetworkLayer.build)
neural_network.set_input([1])
neural_network.evaluate()
