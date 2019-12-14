from math import sqrt

from numpy import (array, unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, arange, outer, meshgrid, dot,
                   logical_and, mean, std, cov, argsort, linspace, transpose,
                   einsum, prod, where, nan, hstack)
from numpy import sum as npsum
from collections import defaultdict, Counter
from warnings import warn
from sys import stdout
from time import time
from datetime import timedelta
import pickle
import os
from sklearn.decomposition import PCA
from PIL import Image
# for unit tests
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from numpy.testing import assert_array_equal
import unittest

"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_order=False):
    """Returns an iterable with the indexes of the samples
    to pick at each iteration of the training."""
    iterations = arange(num_iterations) % data_len
    if random_order:
        random.shuffle(iterations)
    if verbose:
        return _wrap_index__in_verbose(iterations)
    else:
        return iterations


def _wrap_index__in_verbose(iterations):
    """Yields the values in iterations printing the status on the stdout."""
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m-i+1) * (time() - beginning)) / (i+1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        progress += ' - {time_left} left '.format(time_left=time_left)
        stdout.write(progress)


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return sqrt(dot(x, x.T))


def asymptotic_decay(learning_rate, t, max_iter):
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.
    t : int
        current iteration.
    max_iter : int
        maximum number of iterations for the training.
    """
    return learning_rate / (1+t/(max_iter/2))


class MiniSom(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                 decay_function=asymptotic_decay,
                 neighborhood_function='gaussian', random_seed=None):
        """Initializes a Self Organizing Maps.
        A rule of thumb to set the size of the grid for a dimensionality
        reduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.
        E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
        hence a map 8-by-8 should perform well.
        Parameters
        ----------
        x : int
            x dimension of the SOM.
        y : int
            y dimension of the SOM.
        input_len : int
            Number of the elements of the vectors in input.
        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
        learning_rate : initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)
        decay_function : function (default=None)
            Function that reduces learning_rate and sigma at each iteration
            the default function is:
                        learning_rate / (1+t/(max_iterarations/2))
            A custom decay function will need to to take in input
            three parameters in the following order:
            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed
            Note that if a lambda function is used to define the decay
            MiniSom will not be pickable anymore.
        neighborhood_function : function, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map
            possible values: 'gaussian', 'mexican_hat', 'bubble'
        random_seed : int, optional (default=None)
            Random seed to use.
        """
        #sigma不能调的过大
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = random.RandomState(random_seed)

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        # random initialization 初始化所有神经元的权重矩阵
        self._weights = self._random_generator.rand(x, y, input_len)*2-1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)
        self._activation_map = zeros((x, y))
        self._neigx = arange(x)
        self._neigy = arange(y)  # used to evaluate the neighborhood function
        self._decay_function = decay_function

        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and divmod(sigma, 1)[1] != 0:
            warn('sigma should be an integer when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]

    def get_weights(self):
        """Returns the weights of the neural network."""
        return self._weights

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        s = subtract(x, self._weights)  # x - w
        self._activation_map = linalg.norm(s, axis=-1)

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map

    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*pi*sigma*sigma
        ax = exp(-power(self._neigx-c[0], 2)/d)
        ay = exp(-power(self._neigy-c[1], 2)/d)
        return outer(ax, ay)  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c."""
        xx, yy = meshgrid(self._neigx, self._neigy)
        p = power(xx-c[0], 2) + power(yy-c[1], 2)
        d = 2*pi*sigma*sigma
        return exp(-p/d)*(1-2/d*p)

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        ax = logical_and(self._neigx > c[0]-sigma,
                         self._neigx < c[0]+sigma)
        ay = logical_and(self._neigy > c[1]-sigma,
                         self._neigy < c[1]+sigma)
        return outer(ax, ay)*1.

    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-abs(c[0] - self._neigx)) + sigma
        triangle_y = (-abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return outer(triangle_x, triangle_y)

    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self._input_len)
            raise ValueError(msg)

    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)

    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.
        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        #学习率和波及范围都随t增长而衰减
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig)*eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += einsum('ij, ijk->ijk', g, x-self._weights)

    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        self._check_input_len(data)
        q = zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self._weights[self.winner(x)]
        return q

    def random_weights_init(self, data):
        """Initializes the weights of the SOM
        picking random samples from data."""
        self._check_input_len(data)
        it = nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self._random_generator.randint(len(data))
            self._weights[it.multi_index] = data[rand_i]
            it.iternext()

    def pca_weights_init(self, data):
        """Initializes the weights to span the first two principal components.
        This initialization doesn't depend on random processes and
        makes the training process converge faster.
        It is strongly reccomended to normalize the data before initializing
        the weights and use the same normalization for the training data.
        """
        if self._input_len == 1:
            msg = 'The data needs at least 2 features for pca initialization'
            raise ValueError(msg)
        self._check_input_len(data)
        if len(self._neigx) == 1 or len(self._neigy) == 1:
            msg = 'PCA initialization inappropriate:' + \
                  'One of the dimensions of the map is 1.'
            warn(msg)
        pc_length, pc = linalg.eig(cov(transpose(data)))
        pc_order = argsort(-pc_length)
        for i, c1 in enumerate(linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(linspace(-1, 1, len(self._neigy))):
                self._weights[i, j] = c1*pc[pc_order[0]] + c2*pc[pc_order[1]]

    def train(self, data, num_iteration, random_order=False, verbose=False):
        """Trains the SOM.
        Parameters
        ----------
        data : np.array or list
            Data matrix.
        num_iteration : int
            Maximum number of iterations (one iteration per sample).
        random_order : bool (default=False)
            If True, samples are picked in random order.
            Otherwise the samples are picked sequentially.
        verbose : bool (default=False)
            If True the status of the training
            will be printed at each iteration.
        """
        self._check_iteration_number(num_iteration)
        self._check_input_len(data)
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                              verbose, random_order)
        for t, iteration in enumerate(iterations):
            self.update(data[iteration], self.winner(data[iteration]),
                        t, num_iteration)
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
            print(' topographic error:', self.topographic_error(data))

    def train_random(self, data, num_iteration, verbose=False):
        """Trains the SOM picking samples at random from data.
        Parameters
        ----------
        data : np.array or list
            Data matrix.
        num_iteration : int
            Maximum number of iterations (one iteration per sample).
        verbose : bool (default=False)
            If True the status of the training
            will be printed at each iteration.
        """
        self.train(data, num_iteration, random_order=True, verbose=verbose)

    def train_batch(self, data, num_iteration, verbose=False):
        """Trains the SOM using all the vectors in data sequentially.
        Parameters
        ----------
        data : np.array or list
            Data matrix.
        num_iteration : int
            Maximum number of iterations (one iteration per sample).
        verbose : bool (default=False)
            If True the status of the training
            will be printed at each iteration.
        """
        self.train(data, num_iteration, random_order=False, verbose=verbose)

    def distance_map(self):
        """Returns the distance map of the weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours."""
        um = zeros((self._weights.shape[0], self._weights.shape[1]))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                    if (ii >= 0 and ii < self._weights.shape[0] and
                            jj >= 0 and jj < self._weights.shape[1]):
                        w_1 = self._weights[ii, jj, :]
                        w_2 = self._weights[it.multi_index]
                        um[it.multi_index] += fast_norm(w_1-w_2)
            it.iternext()
        um = um/um.max()
        return um

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        self._check_input_len(data)
        a = zeros((self._weights.shape[0], self._weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        self._check_input_len(data)
        error = 0
        for x in data:
            error += fast_norm(x-self._weights[self.winner(x)])
        return error/len(data)

    def topographic_error(self, data):
        """Returns the topographic error computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.
        A sample for which these two nodes are not ajacent conunts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.
        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples."""
        self._check_input_len(data)
        total_neurons = prod(self._activation_map.shape)
        if total_neurons == 1:
            warn('The topographic error is not defined for a 1-by-1 map.')
            return nan

        def are_adjacent(a, b):
            """Gives 0 if a and b are neighbors, 0 otherwise"""
            return not (abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1)

        error = 0
        for x in data:
            self.activate(x)
            activations = self._activation_map
            flat_map = activations.reshape(total_neurons)
            indexes = argsort(flat_map)
            bmu_1 = unravel_index(where(indexes == 0)[0][0],
                                  self._activation_map.shape)
            bmu_2 = unravel_index(where(indexes == 1)[0][0],
                                  self._activation_map.shape)
            error += are_adjacent(bmu_1, bmu_2)
        return error / float(len(data))

    def win_map(self, data):
        """Returns a dictionary wm where wm[(i,j)] is a list
        with all the patterns that have been mapped in the position i,j."""
        self._check_input_len(data)
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap

    def labels_map(self, data, labels):
        """Returns a dictionary wm where wm[(i,j)] is a dictionary
        that contains the number of samples from a given label
        that have been mapped in position i,j.
        Parameters
        ----------
        data : np.array or list
            Data matrix.
        label : np.array or list
            Labels for each sample in data.
        """
        self._check_input_len(data)
        if not len(data) == len(labels):
            raise ValueError('data and labels must have the same length.')
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[self.winner(x)].append(l)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap

"""
data = [[ 0.80,  0.55,  0.22,  0.03],
        [ 0.82,  0.50,  0.23,  0.03],
        [ 0.80,  0.54,  0.22,  0.03],
        [ 0.80,  0.53,  0.26,  0.03],
        [ 0.79,  0.56,  0.22,  0.03],
        [ 0.75,  0.60,  0.25,  0.03],
        [ 0.77,  0.59,  0.22,  0.03]]
som = MiniSom(2, 1, 4, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
som.train_random(data, 100) # trains the SOM with 100 iterations
print(som.winner([ 0.82,  0.50,  0.23,  0.03]))
print(som.winner([ 0.2,  0.59,  0.22,  0.03]))
"""
"""
data = [[ 1,  1,  1,  1],
        [ 1,  1,  1,  1],
        [ 1,  1,  1,  1],
        [ 0,  0,  0,  0],
        [ 0,  0,  0,  0],
        [ 0,  0,  0,  0],
        [ 0,  0,  0,  0]]
som = MiniSom(2, 1, 4, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
som.train_random(data, 100) # trains the SOM with 100 iterations
print(som.winner([ 1,  1,  1,  1]))
print(som.winner([ 0,  0,  0,  0]))
"""
#"""
def dimension_reduction(data, k = 2):
    pca = PCA(n_components=k).fit_transform(data)
    #print(pca)
    #print(pca.shape)
    return pca

# 设置主元个数
k = 20
ary = None
for i in range(2):
    dir_path = "/Users/zhangchen/Documents/课程/神经网络及应用/NN作业用到的材料/face/s{}/".format(i + 1)
    files = os.listdir(dir_path)
    for file_name in files:
        if file_name.endswith("bmp"):
            file_path = dir_path + file_name
            img = Image.open(file_path)
            data = list(img.getdata())
            data = array(data).reshape(-1, 1)
            if ary is None:
                ary = data
            else:
                ary = hstack((ary, data))

feature = dimension_reduction(ary, k)
data = dot(feature.transpose(), ary)
print(ary)
print(data)
train = []

# 50%测试数据
# for i in (list(range(0, 5)) + list(range(10, 15))):
#     train.append(data[:, i].tolist())
#
# print(train)
#
# test = []
# for i in (list(range(5, 10)) + list(range(15, 20))):
#     test.append(data[:, i].tolist())

# 80%测试数据
for i in (list(range(0, 8)) + list(range(10, 18))):
    train.append(data[:, i].tolist())

print(train)

test = []
for i in (list(range(8, 10)) + list(range(18, 20))):
    test.append(data[:, i].tolist())
som = MiniSom(2, 1, k, sigma=0.3, learning_rate=0.5)
som.train_random(train, 100)
for i in test:
    print(som.winner(i))
# print(len(ary[0][0]))
# data = ary[0][:5] + ary[1][:5]
# som = MiniSom(2, 1, len(ary[0][0]), sigma=0.3, learning_rate=0.5)
# for i in ary[0][5:]:
#     print(som.winner(i))
#
# for i in ary[1][5:]:
#     print(som.winner(i))
#"""