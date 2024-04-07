import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_filters=16, filter_size=3, filter_org=1, activation=nn.ReLU(), dense_size=256, dropout=0.2, use_batch_norm=False):
        super(CNN, self).__init__()
        self.input_channels = 3
        self.num_classes = 10

        self.conv_layers = nn.Sequential()
        in_channels = self.input_channels
        for i in range(5):
            self.conv_layers.add_module(f"conv{i+1}", nn.Conv2d(in_channels, num_filters, filter_size))
            if use_batch_norm:
                self.conv_layers.add_module(f"batchnorm{i+1}", nn.BatchNorm2d(num_filters))
            self.conv_layers.add_module(f"activ{i+1}", activation)
            self.conv_layers.add_module(f"maxpool{i+1}", nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = num_filters
            num_filters = int(num_filters * filter_org)

        # Calculate the input size for the dense layers dynamically
        input_size = self._get_conv_output_size((self.input_channels, 224, 224))
        self.dense_layers = nn.Sequential()
        self.dense_layers.add_module(f"linear1", nn.Linear(input_size, dense_size))
        self.dense_layers.add_module(f"activ6", activation)
        self.dense_layers.add_module(f"dropout1", nn.Dropout(dropout))
        self.dense_layers.add_module(f"linear2", nn.Linear(dense_size, self.num_classes))


    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self._forward_dense_layers(x)
        return torch.nn.functional.softmax(x, dim=1)

    def _get_conv_output_size(self, shape):
        # forward a dummy input through the conv layers to get the output size
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output = self._forward_conv_layers(input)
        n_size = output.view(batch_size, -1).size(1)
        return n_size

    def _forward_conv_layers(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def _forward_dense_layers(self, x):
        for layer in self.dense_layers:
          x = layer(x)
        return x