# README: Convolutional Neural Network (CNN) Project Overview

This repository contains Python scripts for training, testing, and visualizing a Convolutional Neural Network (CNN) model using PyTorch. The project includes functionalities for hyperparameter tuning, dataset loading, model training/testing, and visualization of predictions.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Scripts](#scripts)
4. [Functionality](#functionality)
5. [Examples](#examples)
6. [References](#references)

## Installation <a name="installation"></a>

Before running the scripts, ensure you have Python 3.x installed along with the necessary libraries:

- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm
- wandb (optional, for logging)

You can install the required libraries using pip:

```bash
pip install torch torchvision matplotlib numpy tqdm wandb
```

## Usage <a name="usage"></a>

1. Clone this repository to your local machine.
2. Navigate to the repository directory.
3. Run the desired scripts using Python.

## Scripts <a name="scripts"></a>

The repository includes the following scripts:

- `argparser.py`: Contains the argument parser function for customizing the CNN model's architecture.
- `dataloader.py`: Provides functionality for loading datasets for training, validation, and testing.
- `model.py`: Defines the architecture of the CNN model.
- `params_sweep.py`: Performs hyperparameter tuning using the wandb platform.
- `train.py`: Trains and tests the CNN model with specified hyperparameters.
- `predict_and_visualize.py`: Predicts test set samples and visualizes predictions, optionally logging them to wandb.
- `plot_samples_util.py`: Contains utility functions for visualizing test set samples and predictions.

## Functionality <a name="functionality"></a>

- **Dataset Loading**: The `dataloader.py` script provides functionality to load datasets, including preprocessing and data augmentation.
- **Model Architecture**: The `model.py` script defines the architecture of the CNN model with customizable hyperparameters.
- **Training and Testing**: The `train_and_test.py` script trains the CNN model with specified hyperparameters and evaluates its performance on the test set.
- **Hyperparameter Tuning**: The `hyperparameter_tuning.py` script performs hyperparameter tuning using the wandb platform, exploring different configurations.
- **Prediction and Visualization**: The `predict_and_visualize.py` script predicts test set samples using the trained model and visualizes the predictions along with confidence scores. It also logs the visualizations to wandb for further analysis.

## Examples <a name="examples"></a>

### Training and Testing
```bash
python train.py --num_filters 32 --filter_size 5 --filter_org 1 --activation ReLU --dense_size 512 --dropout 0.3 --use_batch_norm
```

### Hyperparameter Tuning
```bash
python params_sweep.py
```

### Visualization and Logging
```bash
python predict_and_visualize.py
```

## References <a name="references"></a>

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- wandb Documentation: https://docs.wandb.ai/