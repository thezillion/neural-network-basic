import numpy as np

def sigmoid(x, deriv=False):
    if (deriv == True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

def correct_out(x, in_input=False):
    if (in_input):
        return 1-np.round(x)
    return np.round(x)

X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    #[1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
])

y = np.array([
    [0],
    [1],
    [1],
    [0],
    #[1],
    [0],
    [0],
    [1],
])

np.random.seed(1)

data_count = len(X)

syn0 = 2*np.random.random((3, data_count))-1
syn1 = 2*np.random.random((data_count, 1))-1

for j in xrange(60000):
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    l2_error = y - l2

    if (j % 10000) == 0:
        print("Error: "+ str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * sigmoid(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * sigmoid(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print("Output after training", l2)

X1 = input("Enter input data: ")
L0 = X1
L1 = sigmoid(np.dot(L0, syn0))
L2 = sigmoid(np.dot(L1, syn1))
print("Output: ", correct_out(L2, X1 in X))
