import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import wandb

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def plot_test_samples(model, test_loader, wandb_log=False, wandb_project=None, wandb_run_name=None):
    classes = ('Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia')

    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # make predictions
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    num_images = len(images)
    fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 3))

    for idx in range(num_images):
        # display image
        ax = axes[idx, 0]
        imshow(images[idx])
        ax.set_title('Ground Truth: {}'.format(classes[labels[idx]]))

        # display prediction
        ax = axes[idx, 1]
        imshow(torchvision.utils.make_grid(images[idx]))
        ax.set_title('Prediction: {}'.format(classes[predicted[idx]]))

        # display confidence scores
        ax = axes[idx, 2]
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(outputs[idx]).detach().numpy()
        ax.bar(classes, probs[0])
        ax.set_title('Confidence Scores')

    plt.tight_layout()
    plt.show()

    if wandb_log:
        if wandb_project is None or wandb_run_name is None:
            raise ValueError("Please provide WandB PROJECT_NAME and RUN_NAME.")

        wandb.init(project=wandb_project, name=wandb_run_name)

        for idx in range(num_images):
            wandb.log({
                "Ground Truth": classes[labels[idx]],
                "Prediction": classes[predicted[idx]],
                "Confidence Scores": {class_name: float(score) for class_name, score in zip(classes, probs[0])},
                "Image": [wandb.Image(images[idx], caption="Ground Truth: {}\nPrediction: {}".format(classes[labels[idx]], classes[predicted[idx]]))]
            })

        wandb.finish()

# Usage:
# plot_test_samples(model, test_loader, wandb_log=True, wandb_project="your_project_name", wandb_run_name="your_run_name")