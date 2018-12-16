import numpy as np

debug = False


class NetworkLayer:

    def __init__(self, layer_id, weights, biases, sigmoid_function):
        self.number_of_nodes = len(biases)
        self.biases = biases
        self.layer_id = layer_id
        self.weights = weights
        self.sigmoid_function = sigmoid_function
        self.z_values = None
        self.activations = None
        self.error = None

    @staticmethod
    def build(upstream_size, layer_id, number_nodes, node_weight_provider,
              node_bias_provider, sigmoid_function):
        weights = np.empty((number_nodes, upstream_size))
        biases = [node_bias_provider() for _ in range(number_nodes)]

        for upstream_node in range(upstream_size):
            weights[:, upstream_node] = node_weight_provider(number_nodes)

        return NetworkLayer(layer_id, weights, biases, sigmoid_function)

    def feed_forward(self, previous_layer_activations):
        weighted_inputs = self.weights.dot(previous_layer_activations)
        self.z_values = np.add(weighted_inputs, self.biases)
        self.activations = [self.sigmoid_function(z_value) for z_value in self.z_values]
        if debug:
            print("For weights: " + str(self.weights) + " and bias " + str(self.biases) + " calculated z: "
                  + str(self.z_values) + " and a: " + str(self.activations))

    def get_number_of_nodes(self):
        return self.number_of_nodes

    def get_z_values(self):
        return self.z_values

    def get_weights(self):
        return self.weights

    def get_activations(self):
        return self.activations

    def set_activations(self, inputs):
        self.activations = inputs

    def apply_deltas(self, weight_deltas, bias_deltas):
        self.weights += weight_deltas
        self.biases += bias_deltas

    def get_biases(self):
        return self.biases
