from neural_network import NeuralNetwork
import math


# sigmoid
def act_func(x):
    return 1 / (1 + math.exp(-x))


# sigmoid derivative
def act_func_der(x):
    return math.exp(x) / (1 + math.exp(x))**2


if __name__ == '__main__':
    predictions = [[0.1, 0.2], [0.01, 0.99]]
    training_data = [[[0.05, 0.1]], [[0.25, 0.74], [0, 0]]]

    sizes = [2, 6, 2]
    nn = NeuralNetwork(act_func, act_func_der, sizes)

    nn.train(10000, 0.5, predictions, training_data)