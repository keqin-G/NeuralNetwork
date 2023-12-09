from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch
from PIL import Image
import os

class MnistData(Dataset):
    def __init__(self, data, transform = None, target_transform = None) -> None:
        super().__init__()
        self.samples = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
class Net(nn.Module):
    def __init__(self, layers) -> None:
        super(Net, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
train_data_dir = '../dataset/mnist/train'
test_data_dir = '../dataset/mnist/test'

train_imgs = []
for label in labels:
    img_dir = os.path.join(train_data_dir, str(label))
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        train_imgs.append((img_path, label))

test_imgs = []
for label in labels:
    img_dir = os.path.join(test_data_dir, str(label))
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        test_imgs.append((img_path, label))


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = MnistData(train_imgs, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

test_dataset = MnistData(test_imgs, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

learning_rate = 0.01
eopches = 20

criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = Net([784, 25, 10]).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train():
    for epoch in range(eopches):
        train_loss = 0
        train_acc = 0

        for im, label in train_loader:
            img = Variable(im.view(im.size(0), -1)).to(device)
            label = Variable(label).to(device)
            out = model(img).to(device)
            loss = criterion(out, label).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc
        
        print(f'epoch: {epoch + 1} / {eopches}, train_loss: {train_loss / len(train_loader)}, train_acc: {train_acc / len(train_loader)}')

train()
# model.load_state_dict(torch.load('model/mnist.pth'))
model.eval()
test_loss = 0
test_acc = 0

for im, label in test_loader:
    img = Variable(im.view(im.size(0), -1)).to(device)
    label = Variable(label).to(device)
    out = model(img).to(device)
    loss = criterion(out, label).to(device)
    test_loss += loss.data
    _, pred = out.max(1)
    num_correct = (pred == label).sum().item()
    acc = num_correct / img.shape[0]
    test_acc += acc

print(f'test_loss: {test_loss / len(test_loader)}, test_acc: {test_acc / len(test_loader)}')
