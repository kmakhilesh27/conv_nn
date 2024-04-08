import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

def plot_test_samples(model, test_loader, wandb_log=False, wandb_project=None, wandb_run_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    
    class_labels = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']

    # Select random samples from the test loader
    sample_indices = np.random.choice(len(test_loader.dataset), size=10, replace=False)

    images_to_log = []

    fig, axes = plt.subplots(10, 3, figsize=(15, 30))
    for i, idx in enumerate(sample_indices):
        image, label = test_loader.dataset[idx]
        image = image.unsqueeze(0).to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = class_labels[predicted.item()]

        # Plot the image
        image_np = (image.cpu().squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        axes[i, 0].imshow(image_np)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"True Label: {class_labels[label]}")

        # Log the predicted label
        axes[i, 1].text(0.5, 0.5, f"Predicted: {predicted_label}", ha='center', va='center', fontsize=12)
        axes[i, 1].axis('off')

        # Log the confidence scores for each class
        confidence_scores = torch.softmax(output, dim=1).squeeze().cpu().detach().numpy()
        axes[i, 2].barh(np.arange(len(class_labels)), confidence_scores)
        axes[i, 2].set_yticks(np.arange(len(class_labels)))
        axes[i, 2].set_yticklabels(class_labels)
        axes[i, 2].set_title('Confidence Scores')
        axes[i, 2].set_xlim(0, 1)

        if wandb_log:
            # Convert image tensor to NumPy array with dtype uint8
            image_np_uint8 = (image.cpu().squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images_to_log.append(wandb.Image(image_np_uint8, caption=f"True Label: {class_labels[label]}, Predicted Label: {predicted_label}"))

    plt.tight_layout()
    plt.show()

    if wandb_log:
        if wandb_project is None or wandb_run_name is None:
            print("Please provide wandb_project and wandb_run_name.")
            return
        else:
            wandb.init(project=wandb_project, name=wandb_run_name)
            wandb.log({"Test Samples": images_to_log})
            wandb.finish()