from functools import reduce

from Link import Link


class Node:
    def __init__(self, node_id, links, bias, sigmoid_function, sigmoid_derivative_function):
        self.node_id = node_id
        self.activation = None
        self.links = links
        self.bias = bias
        self.sigmoid_function = sigmoid_function
        self.sigmoid_derivative_function = sigmoid_derivative_function
        self.z = 0
        self.sigmoid_derivative = 0
        self.node_print("initializing node")

    def node_print(self, message):
        print(self.node_id + " " + message)

    def get_activation(self):
        if self.activation is None:
            reduced = reduce(lambda a, link: a + self.compute_link_value(link), self.links, 0)
            z = reduced + self.bias
            self.node_print("z: " + str(z))
            self.z = z
            self.activation = round(self.sigmoid_function(z), 2)
            self.sigmoid_derivative = self.sigmoid_derivative_function(z)
            self.node_print("calculated my activation " + str(self.activation))
            self.node_print("calculated my sigmoid_derivative " + str(self.sigmoid_derivative))

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

    def get_z_value(self):
        return self.z

    def get_weights(self):
        return [link.weight for link in self.links]

    @staticmethod
    def build(node_id, upstream_layer, weights, bias, sigmoid_function, sigmoid_derivative_function):

        links = []
        if upstream_layer is None:
            return Node(node_id, [], bias, sigmoid_function, sigmoid_derivative_function)
        for i, upstream_node in enumerate(upstream_layer.nodes):
            links.append(Link(upstream_node, weights[i]))

        return Node(node_id, links, bias, sigmoid_function, sigmoid_derivative_function)
