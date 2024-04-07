import torch
from tqdm.auto import tqdm
from model import CNN
from dataloader import load_dataset
from plot_samples_util import plot_test_samples

def predict_test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__=="__main__":
    SAVED_MODEL_PATH = "best_model.pth"
    # Instantiate the best model
    best_model = CNN()
    # Load the saved model
    checkpoint = torch.load(SAVED_MODEL_PATH)
    best_model.load_state_dict(checkpoint)

    _, _, test_loader = load_dataset()
    # Prediction on the test set
    predict_test(best_model, test_loader)
    plot_test_samples(best_model, test_loader, wandb_log=True, wandb_project="CS6910_Assignment_2", wandb_run_name="plot_test_samples")