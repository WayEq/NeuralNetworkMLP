from Utils import *

def test_sigmoid_derivative():
    for i in range (-10,10):
        print(str(i) + ": " + str(Utils.sigmoid_derivative_function(i)))


test_sigmoid_derivative()