import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import shutil

random_state = 42
np.random.seed(random_state)

#path
project_dir = os.getcwd()
raw_data = os.path.join(project_dir, 'train', 'train')
new_dir = os.path.join(project_dir, 'sort_data')
model_path = os.path.join(project_dir, 'model.pt')

#train/test
total_num = len(os.listdir(raw_data)) // 2
random_idx = np.random.permutation(total_num)
train_idx = random_idx[:int(total_num * 0.9)]
test_idx = random_idx[int(total_num * 0.9):]

#dir and label
sub_dirs = ['train', 'test']
animals = ['cats', 'dogs']

for idx, sub_dir in enumerate(sub_dirs):
    dir = os.path.join(new_dir, sub_dir)
    os.makedirs(dir, exist_ok=True)
    for animal in animals:
        animal_dir = os.path.join(dir, animal)
        os.makedirs(animal_dir, exist_ok=True)
        fnames = [f"{animal[:-1]}.{i}.jpg" for i in (train_idx if idx == 0 else test_idx)]
        for fname in fnames:
            src = os.path.join(raw_data, fname)
            dst = os.path.join(animal_dir, fname)
            shutil.copyfile(src, dst)
        print(f"{animal_dir} total images: {len(os.listdir(animal_dir))}")

random_state = 1
torch.manual_seed(random_state)
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)

epochs = 10
batch_size = 4
num_workers = 0
use_gpu = torch.cuda.is_available()

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#load train/test
train_dataset = datasets.ImageFolder(root=os.path.join(new_dir, 'train'), transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

test_dataset = datasets.ImageFolder(root=os.path.join(new_dir, 'test'), transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#cnn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path))
if use_gpu:
    net = net.cuda()
print(net)

#loss fuct
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001)

def train():
    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, train_labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == labels.data).sum()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += train_labels.size(0)

        print('train %d epoch loss: %.3f  acc: %.3f ' % (
            epoch + 1, running_loss / train_total, 100 * train_correct / train_total))
        correct = 0
        test_loss = 0.0
        test_total = 0
        test_total = 0
        net.eval()
        for data in test_loader:
            images, labels = data
            if use_gpu:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_total += labels.size(0)
            correct += (predicted == labels.data).sum()

        print('test  %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, test_loss / test_total, 100 * correct / test_total))

    torch.save(net.state_dict(), model_path)

train()
