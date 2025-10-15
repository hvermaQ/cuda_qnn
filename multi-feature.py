#5-10 features required for encoding to compare the circuit dynamics
#and the effect of depth on the performance of VQC
#consider various chaotic hamiltonians to generate the circuit ansatz

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cudaq
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# =======================
# running routine on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudaq.set_target("nvidia")  # run all quantum circuits on NVIDIA GPU

# =======================
# generate data for classification
x_full, y_full = make_classification(
    n_samples=100, n_features=5, n_informative=5,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

# Normalize features
x_full = (x_full - np.mean(x_full, axis=0)) / np.std(x_full, axis=0)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x_full, y_full, test_size=0.2, random_state=42
)

# Convert labels from {0,1} to {-1,+1}
#y_train = 2 * y_train - 1
#y_test = 2 * y_test - 1

# Plot the data for clarity
plt.figure(figsize=(8, 6))
plt.scatter(x_full[:, 0], x_full[:, 1], c=y_full, cmap='bwr', edgecolors='k')
plt.title("Synthetic Classification Dataset (2 features), one label")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()