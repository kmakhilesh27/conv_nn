import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from model import CustomResNet
from data import get_data_loaders
from train import train_model
from evaluate import evaluate_model, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
# Constants
IMAGE_DIM = 224
BATCH_SIZE = 64
DATA_PATH = "/kaggle/input/inaturalist-12k/inaturalist_12K"
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 10

# Load Data
train_loader, val_loader, test_loader = get_data_loaders(DATA_PATH, IMAGE_DIM, BATCH_SIZE)

# Initialize Model
model = CustomResNet(NUM_CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train Model
train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, device, NUM_EPOCHS)

# Evaluate Model
accuracy, all_labels, all_predictions = evaluate_model(model, test_loader, criterion, device)
print(f'Test Accuracy: {accuracy:.2f}%')

# Plot Confusion Matrix
class_labels = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']
cm = confusion_matrix(all_labels, all_predictions)
plot_confusion_matrix(cm, class_labels)
