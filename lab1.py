import numpy as np
import matplotlib.pyplot as plt

W = np.random.randn(1,2) # 2 random weights (mean 0 variance 1)
B = np.random.randn(1) # random bias

# print(W,B)

# W = weight, X = inputs, B = bias
def sigm(X, W, B): # sigmoid function
    M = 1/(1+np.exp(-(X.dot(W.T)+B)))
    return M

def diff_W(X, Z, Y, B, W): # dW(input, current output, actual output, bias, weight)
    dS = sigm(X,W,B)*(1-sigm(X,W,B))
    dW = (Y-Z)*dS # (y - o)(o(1-o)) # QUESTION: WHY IS THERE NO 2??? Included in learning rate
    return X.T.dot(dW) # .T (tranpose operator) and .dot (dot product)

def diff_B(X, Z, Y, B, W):
    dS = sigm(X,W,B)*(1-sigm(X,W,B))
    dB = (Y-Z)*dS
    return dB.sum(axis=0)

# create 15 two-dimensional samples and their respective outputs

# create training set
X = np.random.randint(2, size = [15,2]) # random inputs (0 or 1)
Y = np.array([X[:,0]|X[:,1]]).T # OR
# Y = np.array([X[:,0]&X[:,1]]).T # AND
# Y = np.array([X[:,0]^X[:,1]]).T # XOR

# create testing set
X_Test = np.random.randint(2, size = [15,2])
Y_Test = np.array([X[:,0]|X[:, 1]]).T # OR
# Y_Test = np.array([X[:,0]&X[:,1]]).T # AND
# Y_Test = np.array([X[:,0]^X[:,1]]).T # XOR

# define learning_rate and epochs iteration number
learning_rate = [0.01, 1, 2, 3]
n_iterations = 1000
leg = []

for i in range(len(learning_rate)): # test with multiple learning_rates
    errors = []
    for epoch in range(n_iterations): # repeat n_iterations to see how errors respond
        output = sigm(X, W, B)
        W += learning_rate[i] * diff_W(X, output, Y, B, W).T # update weight parameter
        B += learning_rate[i] * diff_B(X, output, Y, B, W) # update bias parameter
        errors.append(np.sum(np.sqrt((Y-output)**2), axis = 0))
    plt.plot(range(n_iterations), errors)
    # evaluate model using training set
    leg.append(str(learning_rate[i]), np.sum(np.sqrt((Y_Test-sigm(X_Test, W, B))**2), axis = 0))
    W = np.random.randn(1,2) # 2 random weights (mean 0 variance 1)
    B = np.random.randn(1) # random bias

# evaluate the model on the testing set
test_output = sigm(X_Test, W, B)
print(test_output, Y_Test)
test_error = np.sum(np.sqrt((Y_Test-test_output)**2), axis = 0)

print("Test error:", test_error)

plt.legend(leg)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()