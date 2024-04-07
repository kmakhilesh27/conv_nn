import argparse
import torch.nn as nn

def arg_parser():
    parser = argparse.ArgumentParser(description="ConvNN Parameters")
    parser.add_argument("--num_filters","-nf", type=int, default=16, help="number of filters for the first conv layer")
    parser.add_argument("--filter_size","-fsz", type=int, default=3, help="size of filters for all conv layers")
    parser.add_argument("--filter_org","-fo", type=int, default=1, help="filter organization(multiplier)")
    parser.add_argument("--activation","-ac", type=str, default="ReLU", choices=["ReLU", "GELU", "SiLU", "Mish"], help="activation function")
    parser.add_argument("--dense_size","-dsz", type=int, default=256, help="size of dense layer")
    parser.add_argument("--dropout","-do",  type=float, default=0.2, help="dropout rate")
    parser.add_argument("--use_batch_norm","-bn",action="store_true", help="use batch normalization")
    return parser

def get_activation(activation):
    return getattr(nn, activation)()