import numpy as np
import matplotlib.pyplot as plt

# X is the input, Y is the expected output
X = np.vstack([(np.random.rand(1000,2)*5), (np.random.rand(1000,2)*10)])
Y = np.hstack([([0]*1000), [1]*1000])

# print(X, Y)

# use cross entropy loss function and softmax activation function,
# assign values to Z on the basis of Y as 'one hot vector': 1 or 0
Z = np.zeros((2000,2))

# print(Z)

for i in range(2000):
    Z[i, Y[i]] = 1

print(Z)

# define random intial weights

W1 = np.random.randn(3,2)
B1 = np.random.randn(3)
W2 = np.random.randn(2,3)
B2 = np.random.randn(2)

def Forward(X, W1, B1, W2, B2):
    # first layer
    H = 1/(1+np.exp(-(X.dot(W1.T)+B1)))
    # second layer
    A = H.dot(W2.T)+B2
    # output (softmax operator)
    expA = np.exp(A)
    Output = expA/expA.sum(axis = 1, keepdims = True)
    # return final output and hidden layer 
    return Output, H

def diff_W2(H, Z, Output):
    return H.T.dot(Z-Output)

def diff_W1(X, H, Z, Output, W2):
    dZ = (Z-Output).dot(W2)*H*(1-H)
    return X.T.dot(dZ)

def diff_B2(Z, Output):
    return (Z-Output).sum(axis=0)

def diff_B1(Z, Output, W2, H):
    return ((Z-Output).dot(W2)*H*(1-H)).sum(axis=0)

learning_rate = 1e-3
Error = []

for epoch in range(5000):
    Output, H = Forward(X, W1, B1, W2, B2)
    W2 += learning_rate * diff_W2(H, Z, Output).T
    B2 += learning_rate * diff_B2(Z, Output)
    W1 += learning_rate * diff_W1(X, H, Z, Output, W2).T
    B1 += learning_rate * diff_B1(Z, Output, W2, H)
    Error.append(np.mean(-Z*np.log(Output)))
plt.plot(range(5000), Error)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()