import Utils
import unittest
import numpy as np
import NetworkLayer

class TestNetworkLayer(unittest.TestCase):

    def test_sigmoid_derivative(self):
        for i in range (-10,10):
            print(str(i) + ": " + str(Utils.sigmoid_derivative_function(i)))

    def test_feed_forward(self):
        network_layer = NetworkLayer.NetworkLayer("a", np.array([[.5, .2], [.1,.3]]), np.array([.2,.5]), Utils.sigmoid_function)
        network_layer.feed_forward([1,2])
        values = network_layer.z_values
        np.testing.assert_array_equal(values, np.array([1.1, 1.2]))


class TestNetworkTuner(unittest.TestCase):

    def test_calculate_cost_gradient(self):
        network_layer = NetworkLayer.NetworkLayer("a", np.array([[.5, .2], [.1,.3]]), np.array([.2,.5]), Utils.sigmoid_function)


suite = unittest.TestLoader().loadTestsFromTestCase(TestNetworkLayer)
unittest.TextTestRunner(verbosity=2).run(suite)