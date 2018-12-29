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
        self.vectorized_sigmoid = np.vectorize(self.sigmoid_function)

    @staticmethod
    def build(upstream_size, layer_id, number_nodes, node_weight_provider,
              node_bias_provider, sigmoid_function):
        weights = np.empty((number_nodes, upstream_size))
        biases = np.array([node_bias_provider() for _ in range(number_nodes)])

        for upstream_node in range(upstream_size):
            weights[:, upstream_node] = node_weight_provider(number_nodes)

        return NetworkLayer(layer_id, weights, biases, sigmoid_function)

    def feed_forward(self, previous_layer_activations):
        weighted_inputs = self.weights.dot(previous_layer_activations)
        self.z_values = np.add(weighted_inputs, self.biases)
        self.activations = self.vectorized_sigmoid(self.z_values)
        if debug:
            print("For weights: " + str(self.weights) + " and bias " + str(self.biases) + " calculated z: "
                  + str(self.z_values) + " and a: " + str(self.activations))

    def batch_feed_forward(self, previous_layer_activations):
        weighted_inputs = np.matmul(self.weights,previous_layer_activations)
        self.z_values = np.add(weighted_inputs,self.biases.reshape(self.number_of_nodes,1))

        self.activations = self.vectorized_sigmoid(self.z_values)
        if debug:
            print("For weights: " + str(self.weights) + " and bias " + str(self.biases) + " calculated z: "
                  + str(self.z_values) + " and a: " + str(self.activations))

    def set_activations(self, inputs):
        self.activations = inputs

    def apply_deltas(self, weight_deltas, bias_deltas):
        np_sum = np.sum(weight_deltas, axis=2)
        self.weights += np_sum
        self.biases += np.sum(bias_deltas, axis=1)

    def display(self):
        for i, bias in enumerate(self.biases):
            print(str(self.weights[i][0]) + " bias: " + str(bias))
            for weight in self.weights[i][1:]:
                print(str(weight))
            print("\n")
