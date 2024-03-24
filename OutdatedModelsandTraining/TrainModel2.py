import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np
import os

# Define data directory and other hyperparameters
data_dir = "data/garbage_classification"
batch_size = 32
num_classes = len(os.listdir(data_dir))
input_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20
learning_rate = 0.001
patience = 3

# Define data transformations
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# Split dataset into training and validation sets
num_samples = len(dataset)
split_ratio = 0.8
split = int(split_ratio * num_samples)
indices = list(range(num_samples))
np.random.shuffle(indices)
train_indices, val_indices = indices[:split], indices[split:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# Define the model
model = models.mobilenet_v2(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Linear(1024, num_classes)
)
model = model.to(device)

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)

# Early stopping variables
early_stopping_counter = 0
best_val_loss = float('inf')
best_model_weights = None

# Training loop with tqdm progress bar
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_corrects = 0
    train_total = 0
    
    train_iterator = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', position=0, leave=True)
    
    for inputs, labels in train_iterator:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_corrects += torch.sum(preds == labels.data)
        train_total += labels.size(0)
        
        train_iterator.set_postfix({'Loss': train_loss / train_total, 'Accuracy': train_corrects.double() / train_total})
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_total = 0
    
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        val_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        val_corrects += torch.sum(preds == labels.data)
        val_total += labels.size(0)
    
    val_loss /= val_total
    val_acc = val_corrects.double() / val_total
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / train_total:.4f}, '
          f'Train Acc: {train_corrects.double() / train_total:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = model.state_dict()
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break
    
    # Step the learning rate scheduler
    scheduler.step(val_loss)

# Load the best model weights
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

# Save the model
torch.save(model.state_dict(), 'garbage_classification_model.pth')
