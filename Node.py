from functools import reduce

from Link import Link


class Node:
    def __init__(self, node_id, links, bias, sigmoid_function, sigmoid_derivative_function):
        self.node_id = node_id
        self.activation = None
        self.links = links
        self.bias = bias
        self.debug = False
        self.sigmoid_function = sigmoid_function
        self.sigmoid_derivative_function = sigmoid_derivative_function
        self.z = 0

    def node_print(self, message):
        if (self.debug):
            print(self.node_id + " " + message)

    def get_activation(self):
        if self.activation is None:
            reduced = reduce(lambda a, link: a + self.compute_link_value(link), self.links, 0)
            z = reduced + self.bias
            self.z = z
            self.activation = self.sigmoid_function(z)
            self.node_print("Node (in/b/z/a) " + str(round(reduced,4)) + " / " + str(round(self.bias,4))  + " / " + str(round(z,4)) + " / " + str(round(self.activation,4)))

        return self.activation

    def compute_link_value(self, link):
        link_value = link.weight * link.upstream_node.get_activation()
        self.node_print("Link (a/w/v) " + str(round(link.upstream_node.get_activation(),4)) +" / " + str(round(link.weight,4)) + " / " + str(round(link_value,4)))
        return link_value

    def set_activation(self, activation):
        # self.node_print("Setting activation of input node")
        self.activation = activation

    def get_z_value(self):
        return self.z

    def get_weights(self):
        return [link.weight for link in self.links]

    def get_links(self):
        return self.links

    def get_bias(self):
        return self.bias

    def set_bias(self, bias):
        self.bias = bias

    def clear_activation(self):
        self.activation = None


    @staticmethod
    def build(node_id, upstream_layer, weights, bias, sigmoid_function, sigmoid_derivative_function):

        links = []
        if upstream_layer is None:
            return Node(node_id, [], bias, sigmoid_function, sigmoid_derivative_function)
        for i, upstream_node in enumerate(upstream_layer.nodes):
            links.append(Link(upstream_node, weights[i]))

        return Node(node_id, links, bias, sigmoid_function, sigmoid_derivative_function)
