import datetime
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import toadstool_data_loader
from torch.utils.data import random_split
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.counter = 0

    def __len__(self):
        return len(self.bvp_values)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, f'frame_{idx}.png')
        image = Image.open(img_name)
        self.counter += 1

        bvp_value = self.bvp_values[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(bvp_value, dtype=torch.float, device=self.device) 

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

images_dir = 'toadstool/images/par_2'
data_dir = "toadstool/participants"
single_participant = toadstool_data_loader.load_single_participant(data_dir, 2)
bvp_values = single_participant['BVP']  
model_dir = 'toadstool/BVPmodels'
os.makedirs(model_dir, exist_ok=True)
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_file = os.path.join(model_dir, f"{time} training.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Log the start of training
logging.info('Starting training...')

# Create dataset and dataloader
dataset = BVPDataset(images_dir, bvp_values, transform=transform)
train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): print(f'Using:  {device}')
# Initialize model, optimizer, and loss function
model = BVP_CNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)



load = False
checkpoint_path = os.path.join(model_dir, 'BVP_model_finished.pth')
if os.path.exists(checkpoint_path) and load == True:
    print(f"Loading model and optimizer state from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
else:
    start_epoch = 0
    best_val_loss = float('inf')

# Training loop with early stopping
num_epochs = 100000
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0


for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.unsqueeze(1)  
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)            
            outputs = model(inputs)
            labels = labels.unsqueeze(1) 
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # Get current time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Print statement with current time
    print(f'[{current_time}] Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
          

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        model_name = 'BVP_model_finished.pth'
        model_path = os.path.join(model_dir, model_name)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            logging.info('Early stopping triggered!')
            print('Early stopping!')
            break

logging.info('Training completed.')


# # Load the best model
# model.load_state_dict(torch.load('best_model.pt'))

# # Example prediction
# model.eval()
# with torch.no_grad():
#     input_example = dataset[0][0].unsqueeze(0)
#     prediction = model(input_example)
#     print(f'Predicted BVP value: {prediction.item()}')