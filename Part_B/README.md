# Fine-tuning ResNet50 for Image Classification with PyTorch

This directory contains code for training and evaluating a ResNet50-based image classifier using PyTorch. The model is fine-tuned on the iNaturalist 12K dataset.

## Requirements

- Python >= 3.6
- PyTorch
- torchvision
- scikit-learn
- tqdm
- seaborn
- matplotlib

## Project Structure

The project is organized into several modules:

1. **model.py**: Defines the custom ResNet50 model architecture.
2. **data.py**: Handles dataset loading and transformations.
3. **train.py**: Contains functions for training the model.
4. **evaluate.py**: Provides functions for evaluating the trained model and plotting the confusion matrix.
5. **main.py**: Main script to run the training, evaluation, and visualization.

## Usage

1. Clone the repository:

```
git clone <github-repository-url>
```

2. Install the required dependencies.

3. Download the iNaturalist 12K dataset and place it in the desired location. Update the `DATA_PATH` variable in `main.py` with the path to the dataset.

4. Run the training script:

```
python main.py
```

5. After training, the script will output the test accuracy and display a confusion matrix visualizing the model's performance on the test set.

## Configuration

You can adjust various hyperparameters and configurations in the `main.py` script, such as:

- `IMAGE_DIM`: Image dimensions for resizing and cropping.
- `BATCH_SIZE`: Batch size for training and evaluation.
- `NUM_EPOCHS`: Number of training epochs.
- `LEARNING_RATE`: Learning rate for the optimizer.
- `NUM_CLASSES`: Number of output classes.


## Acknowledgments

- The model architecture is based on the ResNet50 implementation in torchvision.