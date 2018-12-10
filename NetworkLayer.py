class NetworkLayer:

    def __init__(self, layer_id, nodes):
        self.layer_id = layer_id
        self.nodes = nodes
        self.error = None

    @staticmethod
    def build(upstream_layer, layer_id, number_nodes, node_weight_provider, node_bias_provider, sigmoid_function,
              sigmoid_derivative_function, node_builder):
        nodes = []
        upstream_size = 0 if upstream_layer is None else len(upstream_layer.nodes)
        for node_index in range(number_nodes):
            node_id = layer_id + "." + str(node_index)
            node = node_builder(node_id, upstream_layer, node_weight_provider(upstream_size), node_bias_provider(),
                                sigmoid_function, sigmoid_derivative_function, )
            nodes.append(node)
        return NetworkLayer(layer_id, nodes)

    def get_node(self, node_index):
        return self.nodes[node_index]

    def get_nodes(self):
        return self.nodes

    def get_z_values(self):
        return [node.get_z_value() for node in self.nodes]

    def get_weights(self):
        return [node.get_weights() for node in self.nodes]

    def get_activations(self):
        return [node.get_activation() for node in self.nodes]

    def clear_activations(self):
        list(map(lambda node: node.clear_activation(), self.nodes))