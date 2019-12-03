import numpy as np
from math import exp
from math import sin

from random import seed
from random import randrange
from random import random
from csv import reader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


class Neuron:
    def __init__(self, inputs):
        self._inputs = inputs
        self._weights = np.random.random((1, inputs))[0, :]
        self._output = None
        self._delta = None

    def __str__(self):
        s = ""
        s = s + "weights: " + str(self._weights)
        s = s + "\noutput: " + str(self._output)
        s = s + "\ndelta: " + str(self._delta)
        return s

class BPNetwork:
    # Initialize a network
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.network = list()
        hidden_layer = [Neuron(n_inputs + 1) for i in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [Neuron(n_hidden + 1) for i in range(n_outputs)]
        self.network.append(output_layer)

    def __str__(self):
        s = ""
        i = 0
        for layer in self.network:
            s = s + "第" + str(i) + "层：\n"
            i += 1
            for neuron in layer:
                s += str(neuron) + "\n"
        return s

    def activate(self, neuron, inputs):
        weights = neuron._weights
        #print("weights" + str(weights))
        #print("inputs" + str(inputs))
        activation = weights[-1]
        #print(range(len(weights) - 1))
        for i in range(len(weights) - 1):
            #print("value {}  {}".format(weights[i], inputs[i]))
            activation += weights[i] * inputs[i]
        #print("activation" + str(activation))
        return activation

    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    # def forward_propagate(self, row):
    #     inputs = row
    #     for i in range(len(self.network)):
    #         #print("layer" + str(layer))
    #         layer = self.network[i]
    #         new_inputs = []
    #         for neuron in layer:
    #             #print("neuron" + str(neuron))
    #             activation = self.activate(neuron, inputs)
    #             #print("activation" + str(activation))
    #             neuron._output = self.transfer(activation)
    #             new_inputs.append(neuron._output)
    #         inputs = new_inputs
    #     #print("forward_propagate result" + str(inputs))
    #     return inputs

    # def backward_propagate_error(self, expected):
    #     for i in reversed(range(len(self.network))):
    #         layer = self.network[i]
    #         errors = list()
    #         if i != len(self.network) - 1:
    #             for j in range(len(layer)):
    #                 error = 0.0
    #                 for neuron in self.network[i + 1]:
    #                     error += (neuron._weights[j] * neuron._delta)
    #                 errors.append(error)
    #         else:
    #             for j in range(len(layer)):
    #                 neuron = layer[j]
    #                 errors.append(expected - neuron._output)
    #                 #print(errors)
    #         for j in range(len(layer)):
    #             neuron = layer[j]
    #             neuron._delta = errors[j] * self.transfer_derivative(neuron._output)
    #
    # def update_weights(self, row, learning_rate):
    #     for i in range(len(self.network)):
    #         inputs = row[:-1]
    #         if i != 0:
    #             inputs = [neuron._output for neuron in self.network[i - 1]]
    #         for neuron in self.network[i]:
    #             for j in range(len(inputs)):
    #                 neuron._weights[j] += learning_rate * neuron._delta * inputs[j]
    #             neuron._weights[-1] += learning_rate * neuron._delta

    def forward_propagate(self, row):
        inputs = row
        for i in range(len(self.network)):
            #print("layer" + str(layer))
            layer = self.network[i]
            new_inputs = []
            for neuron in layer:
                #print("neuron" + str(neuron))
                activation = self.activate(neuron, inputs)
                #print("activation" + str(activation))
                if i != len(self.network) - 1:
                    neuron._output = self.transfer(activation)
                else:
                    neuron._output = activation
                new_inputs.append(neuron._output)
            inputs = new_inputs
        #print("forward_propagate result" + str(inputs))
        return inputs

    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron._weights[j] * neuron._delta)
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected - neuron._output)
                    #print(errors)
            for j in range(len(layer)):
                neuron = layer[j]
                if i != len(self.network) - 1:
                    neuron._delta = errors[j] * self.transfer_derivative(neuron._output)
                else:
                    neuron._delta = errors[j]

    def update_weights(self, row, learning_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron._output for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron._weights[j] += learning_rate * neuron._delta * inputs[j]
                neuron._weights[-1] += learning_rate * neuron._delta

    # Train a network for a fixed number of epochs
    def train_network(self, train, l_rate, n_epoch):
        for epoch in range(n_epoch):
            #print("epoch" + str(epoch))
            for row in train:
                #print("row" + str(row))
                outputs = self.forward_propagate(row)
                #print("============")
                #print(outputs)
                #expected = [0 for i in range(n_outputs)]
                #expected[row[-1]] = 1
                expected = row[-1]
                self.backward_propagate_error(expected)
                self.update_weights(row, l_rate)

    # Make a prediction with a network
    def predict(self, row):
        outputs = self.forward_propagate(row)
        #return outputs.index(max(outputs))
        return outputs


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    #n_outputs = len(set([row[-1] for row in train]))
    n_outputs = 1
    network = BPNetwork(n_inputs, n_hidden, n_outputs)
    #print(network)
    network.train_network(train, l_rate, n_epoch)
    predictions = list()
    for row in train:
        prediction = network.predict(row)
        #print(row)
        #print(prediction)
        predictions.append(prediction[0])
    return(predictions)

l_rate = 0.005
n_epoch = 1000
n_hidden = 10

train = []

x = np.linspace(-3,3,600)
y = []
for i in x:
    y.append(sin(i))
    train.append([i, sin(i)])

y_pred = back_propagation(train, l_rate, n_epoch, n_hidden)


plt.plot(x, y, '-o', label='true')
plt.plot(x, y_pred, '-o', label='BP-Net')
plt.legend()

plt.tight_layout()
plt.show()

"""
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Test Backprop on Seeds dataset
seed(1)
# load and prepare data
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
"""