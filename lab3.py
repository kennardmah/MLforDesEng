import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train = datasets.MNIST("", train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

# three hidden layers of 64, 784 total input (28x28 pixels)
# output of 10 units
# why 64?

n = 128

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, n) # creating a layer of n neurons with 28*28 inputs
        self.fc2 = nn.Linear(n, n) # nn.Linear is an affine trasformation
        self.fc3 = nn.Linear(n, n)
        self.fc4 = nn.Linear(n, n)
        self.fc5 = nn.Linear(n, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x)) # 0.103 (64 neurons)
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        # x = torch.sigmoid(self.fc4(x)) # better accuracy without fourth layer
        # x = F.relu(self.fc1(x)) # 0.118 (64 neurons)
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = torch.tanh(self.fc1(x)) # 0.411 (64 neurons), 0.47 (128 neurons), 0.353 (200 neurons)
        # x = torch.tanh(self.fc2(x))
        # x = torch.tanh(self.fc3(x))
        # x = torch.tanh(self.fc4(x)) # 0.442 (128 neurons, 4 hidden layers)
        x = self.fc5(x)

        # computes to softmax function of given input
        return F.softmax(x, dim = 1)

# create instance of neural network
net = Net()

# sets up optimisation method for backpropagation alg
# optimiser = optim.SGD(net.parameters(), lr = 0.001) # SGD (Stoachastic Gradient Descent)
optimiser = optim.Adam(net.parameters(), lr = 0.001) # Adam (increases Accuracy drastically)

Epochs = 3 # number of training epochs

# iterate over the training data (trainset)
for epoch in range(Epochs):
    for data in trainset:
        X, y = data # assign input (X) and labels (y)
        net.zero_grad() # set gradiant stored to zero to reset gradient value for each iteration
        output = net.forward(X.view(-1,28*28)) # transform 2 dimensional tensor (28x28 matrix) input to 1 dimension (784 vector)
        loss = F.nll_loss(output, y) # loss function (cross-entropy sicne we are working w classifier)
        loss.backward() # compute gradient wrt loss function over each parameter of the network (must set gradient to 0, line 46)
        optimiser.step() # update parameters of the network according to the optimisation alg and gradient stored within each variable

correct, total = 0, 0

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net.forward(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print("Accuracy: ", round(correct/total, 3))