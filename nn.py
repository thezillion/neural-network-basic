import numpy as np

class Network:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.syn0 = None
        self.syn1 = None

    def sigmoid(self, x, deriv=False):
        if (deriv == True):
            return x*(1-x)

        return 1/(1+np.exp(-x))

    def train(self):
        np.random.seed(1)

        data_count = len(X)
        layer_size = len(X[0])

        syn0 = 2*np.random.random((layer_size, data_count))-1
        syn1 = 2*np.random.random((data_count, 1))-1

        for j in xrange(60000):
            l0 = self.X
            l1 = self.sigmoid(np.dot(l0, syn0))
            l2 = self.sigmoid(np.dot(l1, syn1))
            l2_error = y - l2

            if (j % 10000) == 0:
                print("Error: "+ str(np.mean(np.abs(l2_error))))

            l2_delta = l2_error * self.sigmoid(l2, deriv=True)

            l1_error = l2_delta.dot(syn1.T)

            l1_delta = l1_error * self.sigmoid(l1, deriv=True)

            syn1 += l1.T.dot(l2_delta)
            syn0 += l0.T.dot(l1_delta)

        print("Output after training", l2)
        self.syn0 = syn0
        self.syn1 = syn1

    def test(self, X1):
        L0 = X1
        L1 = self.sigmoid(np.dot(L0, self.syn0))
        L2 = self.sigmoid(np.dot(L1, self.syn1))
        print("Output: ", L2)

"""
Usage
X = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    #[1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
]

y = [
    [0],
    [1],
    [1],
    [0],
    #[1],
    [0],
    [0],
    [1],
]

network = Network(X, y)
network.train()
network.test([1, 0, 0])
"""
