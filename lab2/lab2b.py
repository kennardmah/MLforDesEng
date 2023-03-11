import numpy as np
import matplotlib.pyplot as plt

# X is the input, Y is the expected output
X = np.vstack([(np.random.rand(1000,2)*5), (np.random.rand(1000,2)*10)])
Y = np.hstack([([0]*1000), [1]*1000])

# use cross entropy loss function and softmax activation function,
# assign values to Z on the basis of Y as 'one hot vector': 1 or 0
Z = np.zeros((2000,2))

for i in range(2000):
    Z[i, Y[i]] = 1

# define random intial weights
W1 = np.random.randn(3,2)
B1 = np.random.randn(3)
W2 = np.random.randn(2,3)
B2 = np.random.randn(2)

def Forward(X, W1, B1, W2, B2):
    H = 1/(1+np.exp(-(X.dot(W1.T)+B1)))
    # second layer
    A = H.dot(W2.T)+B2
    # output (softmax operator)
    expA = np.exp(A)
    Y = expA/expA.sum(axis = 1, keepdims = True)
    # return final output and hidden layer 
    return Y, H

def diff_W2(H, Z, Y):
    return H.T.dot(Z-Y)

def diff_W1(X, H, Z, Y, W2):
    dZ = (Z-Y).dot(W2)*H*(1-H)
    return X.T.dot(dZ)

def diff_B2(Z, Y):
    return (Z-Y).sum(axis=0)

def diff_B1(Z, Y, W2, H):
    return ((Z-Y).dot(W2)*H*(1-H)).sum(axis=0)

learning_rate = 0.001
Error = []

for epoch in range(5000):
    Output, H = Forward(X, W1, B1, W2, B2)
    W2 += learning_rate * diff_W2(H, Z, Output).T
    B2 += learning_rate * diff_B2(Z, Output)
    W1 += learning_rate * diff_W1(X, H, Z, Output, W2).T
    B1 += learning_rate * diff_B1(Z, Output, W2, H)
    # cross-entropy function
    Error.append(np.mean(-Z*np.log(Output)))
plt.plot(range(5000), Error)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()