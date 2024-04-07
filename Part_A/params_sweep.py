import wandb
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from model import CNN
from dataloader import load_dataset

SAVED_MODEL_PATH = "best_model.pth"
PROJ_NAME = "CS6910_Assignment_2"
best_val_acc = 0.0

def getRunName(config):
    run_name = "n_filters_{}_filter_size_{}_filter_org_{}_activ_{}_dense_size_{}_b_norm_{}_dropout_{}_lr_{}_aug_{}".format(
        config['num_filters'],
        config['filter_size'],
        config['filter_org'],
        config['activation'],
        config['dense_size'],
        config['use_batch_norm'],
        config['dropout'],
        config['learning_rate'],
        config['augment_data'])
    return run_name

def params_sweep():
    global best_val_acc, SAVED_MODEL_PATH, PROJ_NAME
    wandb.init(project= PROJ_NAME,config = wandb.config)
    config = wandb.config
    wandb.run.name = getRunName(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNN(config.num_filters, config.filter_size, config.filter_org, getattr(nn, config.activation)(),
                config.dense_size, config.dropout, config.use_batch_norm)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loader, val_loader, _ = load_dataset(augment_data=config.augment_data)

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = (100 * correct_val / total_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVED_MODEL_PATH)  # save the best model


        print(f"Epoch {epoch+1}/{config.epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {(100*correct_train/total_train):.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {(100*correct_val/total_val):.2f}%")
        wandb.log({"epoch": epoch+1,
                   "train_loss": train_loss/len(train_loader),
                   "train_acc": (100*correct_train/total_train),
                   "val_loss": val_loss/len(val_loader),
                   "val_acc": (100*correct_val/total_val)
                   })

if __name__ == "__main__":

    sweep_config = {
        "name": "inaturalist-hyper-sweep",
        "method": "bayes",
        "metric": {"name": "val_acc", "goal": "maximize"},
        "parameters": {
            "num_filters": {"values": [32, 48, 64]},
            "filter_size": {"values": [3,4,5]},
            "filter_org" : {"values": [1,2,0.5]},
            "activation": {"values": ["ReLU", "GELU", "SiLU", "Mish"]},
            "dense_size": {"values": [256,512,1024]},
            "use_batch_norm": {"values": [True, False]},
            "dropout": {"values": [0.2, 0.3]},
            "epochs": {"value": 5},
            "learning_rate": {"values": [0.001, 0.01]},
            "augment_data": {"values": [True, False]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project = PROJ_NAME)
    wandb.agent(sweep_id, function=params_sweep, count=20)
    wandb.finish()