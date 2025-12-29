# Code/analysis/complexity.py

import torch
from Code.model_B.cnn import SimpleCNN, DeepCNN


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_complexity():
    simple_cnn = SimpleCNN()
    deep_cnn = DeepCNN()

    return {
        "SimpleCNN": count_parameters(simple_cnn),
        "DeepCNN": count_parameters(deep_cnn),
    }
def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_complexity():
    """
    Return parameter counts for all CNN models.
    """
    simple_cnn = SimpleCNN()
    deep_cnn = DeepCNN()

    complexity = {
        "SimpleCNN": count_parameters(simple_cnn),
        "DeepCNN": count_parameters(deep_cnn),
    }

    return complexity