from functools import reduce

from scipy.stats import logistic

from Link import Link


class Node:
    def __init__(self, node_id, links, bias, sigmoid_function):
        self.node_id = node_id
        self.activation = None
        self.links = links
        self.bias = bias
        self.sigmoid_function = sigmoid_function
        self.z = 0
        self.node_print("initializing node")

    def node_print(self, message):
        print(self.node_id + " " + message)

    def get_activation(self):
        if self.activation is None:
            reduced = reduce(lambda a, link: a + self.compute_link_value(link), self.links, 0)
            z = reduced + self.bias
            self.node_print("before sigmoid: " + str(z))
            self.z = z
            self.activation = round(self.sigmoid_function(z), 2)
            self.node_print("calculated my activation " + str(self.activation))

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

    # def calculate_error(self, cost_derivitive):
        # figure out how much tweaking z affects cost
        # zderivitive = get_z_derivitive()

    @staticmethod
    def build(node_id, upstream_layer, weights, bias, sigmoid_function):

        links = []
        if upstream_layer is None:
            return Node(node_id, [], bias, sigmoid_function)
        for i, upstream_node in enumerate(upstream_layer.nodes):
            links.append(Link(upstream_node, weights[i]))

        return Node(node_id, links, bias, sigmoid_function)