import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cpu")


class ImageReader(nn.Module):
    def __init__(self, num_classes=10):
        super(ImageReader, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))        
        x = self.fc2(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        outputs = model(data)
        loss = F.cross_entropy(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = F.cross_entropy(outputs, target, reduction='sum')
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    avg_loss = test_loss / total
    accuracy = 100.0 * correct / total
    print(f'Average Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')




def main():
    batch_size = 64
    lr = 0.001
    num_epochs = 5
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)
    
    model = ImageReader(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    
    torch.save(model.state_dict(), 'image_reader.pth')
    print("Training Done!")


if __name__ == '__main__':
	main()