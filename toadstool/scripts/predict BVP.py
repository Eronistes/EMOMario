import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import toadstool_data_loader

# Define your CNN model for predicting BVP values
class BVP_CNN(nn.Module):
    def __init__(self):
        super(BVP_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 10 * 10, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)  # Output layer for BVP prediction

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 10 * 10)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class BVPDataset(Dataset):
    def __init__(self, images_dir, bvp_values, transform=None):
        """
        Args:
            images_dir (string): Directory with all the images.
            bvp_values (list or array): List or array of BVP values.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.bvp_values = bvp_values
        self.transform = transform

    def __len__(self):
        return len(self.bvp_values)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, f'frame_{idx}.png')
        image = Image.open(img_name).convert('L')  # Convert image to grayscale
        bvp_value = self.bvp_values[idx]

        if self.transform:
            image = self.transform(image)

        return image, bvp_value

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

images_dir = 'toadstool/images/par_0'
data_dir = "toadstool/participants"
single_participant = toadstool_data_loader.load_single_participant(data_dir, 0)
bvp_values = single_participant['BVP']  # Replace with your actual list of BVP values


# Create dataset and dataloader
dataset = BVPDataset(images_dir, bvp_values, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, optimizer, and loss function
model = BVP_CNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}')

