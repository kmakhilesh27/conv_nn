import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_path, image_dim, batch_size):
    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "val")

    common_transform = transforms.Compose([
        transforms.Resize(image_dim),
        transforms.CenterCrop(image_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(train_dir, transform=common_transform)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    test_dataset = ImageFolder(test_dir, transform=common_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
