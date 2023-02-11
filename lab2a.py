import numpy as np
import matplotlib.pyplot as plt

# why different dimensions of weights?
W1 = np.random.randn(3,2) 
B1 = np.random.randn(3)
W2 = np.random.randn(1,3)
B2 = np.random.randn(1)
# print(W1, B1, W2, B2)

def sigm(X, W, B):
    M = 1/(1+np.exp(-(X.dot(W.T)+B)))
    return M

def Forward(X, W1, B1, W2, B2):
    # first layer (hidden layer)
    H = sigm(X, W1, B1)
    # second layer (final layer)
    Y = sigm(H,W2,B2)
    return Y, H

def diff_B2(Z, Y):
    dB = (Z-Y)*Y*(1-Y)
    return dB.sum(axis=0)

def diff_W2(H, Z, Y):
    dW = (Z-Y)*Y*(1-Y)
    return H.T.dot(dW)

def diff_B1(Z, Y, W2, H):
    return ((Z-Y).dot(W2)*Y*(1-Y)*H*(1-H)).sum(axis=0)

def diff_W1(X, H, Z, Y, W2):
    dZ = (Z-Y).dot(W2)*Y*(1-Y)*H*(1-H)
    return X.T.dot(dZ)

# training data (X is the input values and Z is the expected output for XOR)
X = np.random.randint(2, size=[50,2])
Z = np.array([X[:,0]^X[:, 1]]).T

learning_rate_set = [0.01, 0.1, 1]
epoch_val = 5000
leg = []
for i in range(len(learning_rate_set)):
    learning_rate = learning_rate_set[i]
    leg.append(learning_rate)
    W1 = np.random.randn(3,2) 
    B1 = np.random.randn(3)
    W2 = np.random.randn(1,3)
    B2 = np.random.randn(1)
    Accuracy_set = []
    for epoch in range(epoch_val):
        Y, H = Forward(X, W1, B1, W2, B2)

        W2 += learning_rate * diff_W2(H,Z,Y).T
        B2 += learning_rate * diff_B2(Z,Y)
        W1 += learning_rate * diff_W1(X,H,Z,Y,W2).T
        B1 += learning_rate * diff_B1(Z, Y, W2, H)
        if not epoch %50:
            Accuracy = 1 - np.mean((Z-Y)**2)
            print('Epoch: ', epoch, ' Accuracy: ', Accuracy)
        Accuracy_set.append(1-np.mean((Z-Y)**2))
    plt.plot(range(epoch_val), Accuracy_set)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(leg)
plt.show()