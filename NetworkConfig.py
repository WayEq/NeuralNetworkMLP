from dataclasses import dataclass


@dataclass
class NetworkConfig:
    nodes_per_layer: []
    node_weight_provider: lambda number_of_weights: []
    node_bias_provider: lambda _: int
    sigmoid_function: lambda x: float
    sigmoid_derivative_function: lambda x: float
