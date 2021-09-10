#%%

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

#check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

#need to download the dataset form internet first
mnist_train = torchvision.datasets.MNIST('/mnt/ea25417a-4bfa-4bd9-b932-9c8fa4a1b108/books/mnist',
                                         download=True, train=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,))])
                                         )
mnist_train_loader = torch.utils.data.DataLoader(mnist_train,
                                          batch_size=64,
                                          shuffle=True)
mnist_test = torchvision.datasets.MNIST('/mnt/ea25417a-4bfa-4bd9-b932-9c8fa4a1b108/books/mnist',
                                         download=True, train=False,
                                        transform=torchvision.transforms.Compose(
                                             [torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                             ])
                                        )

mnist_test_loader = torch.utils.data.DataLoader(mnist_test,
                                          batch_size=64,
                                          shuffle=True)

'''train_features, train_labels = next(iter(mnist_train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")'''

def Model():
    model_fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128,10),
        nn.LogSoftmax())
    return model_fc

model = Model()
model.to(device)
optimizer = optim.SGD(model.parameters(),lr=0.01)
loss_fn = nn.NLLLoss().cuda()

n_epochs = 20
for epoch in range(n_epochs):
    for img, label in mnist_train_loader:
        img,label = img.to(device), label.to(device)
        batch_size = img.shape[0]
        out = model(img.view(batch_size,-1))
        loss = loss_fn(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))


correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in mnist_test_loader:
        imgs,labels = imgs.to(device), labels.to(device)
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
    print("Accuracy: %f", correct / total) ## 9722/10000
