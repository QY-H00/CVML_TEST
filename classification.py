import math
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR

# Let torch to estimate if using GPU or not. GPU is much faster than CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10000 # size of batch per round during training, require 2d memory
INPUT_CHANNEL = 3 # RGB
PHOTO_SIZE = 32 # 32 * 32 photo

# Hpyerparameter used for training
EPOCHS = 120 # number of rounds for training
interval_size = 1 # the number of epochs between two points
learning_rate_first = 0.01
learning_rate_second = 0.001
learning_rate_third = 0.0001
alpha = 0.1
beta = 1
eta = 0
A = -4.
momentum = 0.9
weight_decay = 0.0001

# Plot the line
#定义两个数组
Loss_list = []
Accuracy_list = []
Loss_list_2 = []
Accuracy_list_2 = []

#我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
x1 = range(0, EPOCHS)
x2 = range(0, EPOCHS)
y1 = Accuracy_list
y2 = Loss_list

x1_2 = range(0, EPOCHS)
x2_2 = range(0, EPOCHS)
y1_2 = Accuracy_list_2
y2_2 = Loss_list_2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dicts = []
train_X = torch.Tensor()
train_Y = torch.Tensor()
test_X = torch.Tensor()
test_Y = torch.Tensor()

# Transform the data in the format of 10000 * 3 * (32 * 32)
# {data: ____, labels: ____ } 
# where data is a 10000 * 3072 numpy array if uint8
# labels is a list of 10000 numbers in the range 0 - 9
# Change data_batch_i into the dicts[i], integrate to one tensor
for i in range(0, 5):
    dicts.append(unpickle("data/data_batch_" + str(i + 1)))
    train_x = dicts[i][b'data']
    train_x = train_x.reshape(BATCH_SIZE, INPUT_CHANNEL, PHOTO_SIZE, PHOTO_SIZE)
    train_x = torch.from_numpy(train_x)
    train_X = torch.cat((train_X, train_x), 0)
    train_y = dicts[i][b'labels']
    prob_y = np.random.choice(9, len(train_y))
    train_y = [random.randint(0, 9) if prob_y[i] < int(eta * 10) else train_y[i] for i in range(len(train_y))]
    train_y = np.array(train_y)
    train_y = torch.from_numpy(train_y)
    train_Y = torch.cat((train_Y, train_y), 0)

# Change test_batch to tensor
dict_test = unpickle("data/test_batch")
test_X = dict_test[b'data']
test_X = test_X.reshape(BATCH_SIZE, INPUT_CHANNEL, PHOTO_SIZE, PHOTO_SIZE)
test_X = torch.from_numpy(test_X).float()
test_Y = dict_test[b'labels']
test_Y = np.array(test_Y)
test_Y = torch.from_numpy(test_Y)

# Load dataset
train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=False)
test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1st: 3 input channel, 64 output channel, and kernel with size 3
        self.conv11 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.bn12 = nn.BatchNorm2d(64)
        # 2nd: 64 input channel, 128 output channel, and kernel with size 3
        self.conv21 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.bn22 = nn.BatchNorm2d(128)
        # 3rd: 128 input channel, 196 output channel, and kernel with size 3
        self.conv31 = nn.Conv2d(128, 196, 3, padding=1)
        self.conv32 = nn.Conv2d(196, 196, 3, padding=1)
        self.bn31 = nn.BatchNorm2d(196)
        self.bn32 = nn.BatchNorm2d(196)
        # 4th: fully connected layer: 3136 ->  256
        self.fc1 = nn.Linear(3136, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        # 5thL fully connected later: 256 -> 10
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # This is just BATCH_SIZE
        in_size = x.size(0)
        # 1st: conv1 * 2
        out = self.conv11(x)
        out = self.bn11(out)
        out = F.relu(out)
        out = self.conv12(out)
        out = self.bn12(out)
        out = F.relu(out)
        # 2nd: pool1
        out = torch.max_pool2d(out, 2, 2)
        # 3rd: conv2 * 2
        out = self.conv21(out)
        out = self.bn21(out)
        out = F.relu(out)
        out = self.conv22(out)
        out = self.bn22(out)
        out = F.relu(out)
        # 4th: pool2
        out = torch.max_pool2d(out, 2, 2)
        # 5th: conv3 * 2
        out = self.conv31(out)
        out = self.bn31(out)
        out = F.relu(out)
        out = self.conv32(out)
        out = self.bn32(out)
        out = F.relu(out)
        # 6th: pool3
        out = torch.max_pool2d(out, 2, 2)
        # change to BATCH_SIZE * ()
        out = out.view(in_size, -1)
        # 7th: fc1
        out = self.fc1(out)
        out = self.bn_fc1(out)
        out = F.relu(out)
        # 8th: fc2 + softmax
        out = self.fc2(out)
        # out = F.softmax(out, dim=1)
        return out

def reverse_cross_entropy(output, target):
    target = target.reshape(target.shape[0], 1)
    target = torch.zeros(target.shape[0], 10).to(DEVICE).scatter_(1,target,1)
    target = torch.where(target == 0,
                         math.exp(A), 
                         target.double())
    output = F.softmax(output, dim=1)
    return -torch.sum(-torch.sum(-output * torch.log(target), dim=1)) / target.shape[0] 

def custom_loss(output, target, alpha, beta):
    return alpha * F.cross_entropy(output, target) + beta * reverse_cross_entropy(output, target)

# Define the model
model = ConvNet().to(DEVICE)
model_2 = ConvNet().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=learning_rate_first, momentum=momentum, weight_decay=weight_decay)
optimizer_2 = optim.SGD(model_2.parameters(), lr=learning_rate_first, momentum=momentum, weight_decay=weight_decay)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 40:
            optimizer.param_groups[0]['lr'] = learning_rate_second
        elif batch_idx >= 80:
            optimizer.param_groups[0]['lr'] = learning_rate_third
        # transfer data and target to be computed on the device
        data, target = data.to(device), target.long().to(device)
        # clear the gradient in each iteration
        optimizer.zero_grad()
        output = model(data)
        # If we use CrossEntropyLoss here, then we don't need the softmax in the forward() function
        loss = F.cross_entropy(output, target)
        # loss = custom_loss(output, target, alpha, beta)
        loss.backward()
        optimizer.step()
        # Todo: Find the meaning of each parameter
        if (batch_idx + 1) % 25 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    if (epoch % 1 == 0):
        Loss_list.append(loss / (len(train_dataset)))

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.long().to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target)
            # test_loss += torch.sum(custom_loss(output, target, alpha, beta))
            pred = output.max(1, keepdim=True)[1] # Find the index of biggest probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if (epoch % 1 == 0):
            Accuracy_list.append(100. * correct / len(test_loader.dataset))

def train_2(model, device, train_loader, optimizer, epoch):
    model_2.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 40:
            optimizer_2.param_groups[0]['lr'] = learning_rate_second
        elif batch_idx >= 80:
            optimizer_2.param_groups[0]['lr'] = learning_rate_third
        # transfer data and target to be computed on the device
        data, target = data.to(device), target.long().to(device)
        # clear the gradient in each iteration
        optimizer_2.zero_grad()
        output = model_2(data)
        # If we use CrossEntropyLoss here, then we don't need the softmax in the forward() function
        # loss = F.cross_entropy(output, target)
        loss = custom_loss(output, target, alpha, beta)
        loss.backward()
        optimizer_2.step()
        # Todo: Find the meaning of each parameter
        if (batch_idx + 1) % 25 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    if (epoch % interval_size == 0):
        Loss_list_2.append(loss / (len(train_dataset)))

def test_2(model, device, test_loader, epoch):
    model_2.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.long().to(device)
            output = model_2(data)
            # test_loss += F.cross_entropy(output, target)
            test_loss += torch.sum(custom_loss(output, target, alpha, beta))
            pred = output.max(1, keepdim=True)[1] # Find the index of biggest probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if (epoch % interval_size == 0):
            Accuracy_list_2.append(100. * correct / len(test_loader.dataset))


for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader, epoch)
    train_2(model_2, DEVICE, train_loader, optimizer_2, epoch)
    test_2(model_2, DEVICE, test_loader, epoch)

plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-', label="With Cross Entropy Loss")
plt.plot(x1_2, y1_2, 'o-', label="With Robust Loss")
plt.title('When eta =' + str(eta))
plt.ylabel('Test accuracy')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-', label="cross entropy loss")
plt.plot(x2_2, y2_2, '.-', label="robust loss function")
plt.xlabel('EPOCHS')
plt.ylabel('Train loss')
plt.legend()
plt.show()
plt.savefig("eta =" + str(eta) + ".jpg")


        



