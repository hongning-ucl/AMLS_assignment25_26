# Code/model_A/features.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_features(images, use_pca=False, n_components=50):
    """
    images: (N, 28, 28)
    return: (N, D)
    """
    return images.reshape(images.shape[0], -1)