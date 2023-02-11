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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64) # creating a layer of 64 neurons with 28*28 inputs
        self.fc2 = nn.Linear(64, 64) # nn.Linear is an affine trasformation
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x) # why not sigmoid?

        # computes to softmax function of given input
        return F.softmax(x, dim = 1)

# create instance of neural network
net = Net()

# sets up optimisation method for backpropagation alg
optimiser = optim.SGD(net.parameters(), lr = 0.001) # SGD (Stoachastic Gradient Descent)
Epochs = 3 # number of training epochs

# iterate over the training data (trainset)
for epoch in range(Epochs):
    for data in trainset:
        print(data)
        X, y = data # assign input (X) and labels (y)
        net.zero_grad() # set gradiant stored to zero to reset gradient value for each iteration
        output = net.forward(X.view(-1,28*28)) # transform 2 dimensional tensor (28x28 matrix) input to 1 dimension (784 vector)
        loss = F.nll_loss(output, y) # loss function (cross-entropy sicne we are working w classifier)
        loss.backward() # compute gradient wrt loss function over each parameter of the network (must set gradient to 0, line 46)
        optimiser.step() # update parameters of the network according to the optimisation alg and gradient stored within each variable