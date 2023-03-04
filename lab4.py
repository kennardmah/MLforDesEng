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
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

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