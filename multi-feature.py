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
import math

#n features mapped to n qubits
#however only one label
#what to measure: sum of the z expectation values of all qubits
n_qubits = 4 #also the number of features

hamiltonian = cudaq.spin.z(0)  # measure the z of qubit 0 for label prediction
for i in range(1, n_qubits):
    hamiltonian = cudaq.spin.z(i)  # measure the z of qubit i for label prediction

# running routine on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudaq.set_target("nvidia")  # run all quantum circuits on NVIDIA GPU

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
plt.title("Synthetic Classification Dataset (5 features), one label")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

#consider various chaotic hamiltonians to generate the circuit ansatz
#for now a single Hamiltonian with tunable chaos -- lambda

# Helper: bounded reparameterization of variational angles
#to create a certain ratio of lambda to J
def bounded_thetas(raw_params, theta0=0.1, eps_max=0.2):
    """
    Maps unconstrained parameters p_k to bounded θ_k.
    θ_k ∈ [θ0(1 - eps_max), θ0(1 + eps_max)] via tanh transform.
    """
    return theta0 * (1.0 + eps_max * np.tanh(np.array(raw_params)))

#main kernel to tune chaos
@cudaq.kernel
def XXZ_model(lambda_param: float,
                           n_qubits: int,
                           N_trot: int,
                           thetas: list[float],
                           h: list[float]):
    """
    thetas: one variational angle per Trotter step, shared across XX, YY, ZZ
    h: local fields (can be random)
    lambda_param: chaos control
    """
    q = cudaq.qvector(n_qubits)

    for k in range(N_trot):
        theta = thetas[k]  # shared angle for this Trotter step

        for i in range(n_qubits - 1):
            # XX coupling
            cx(q[i], q[i+1])
            rx(2 * theta, q[i+1])
            cx(q[i], q[i+1])

            # YY coupling
            ry(np.pi / 2, q[i])
            ry(np.pi / 2, q[i+1])
            cx(q[i], q[i+1])
            rx(2 * theta, q[i+1])
            cx(q[i], q[i+1])
            ry(-np.pi / 2, q[i])
            ry(-np.pi / 2, q[i+1])

            # ZZ coupling
            cx(q[i], q[i+1])
            rz(2 * theta, q[i+1])
            cx(q[i], q[i+1])

        # Local Z fields — break integrability, control chaos
        # no variational angle
        for i in range(n_qubits):
            rz(2 * lambda_param * h[i], q[i])


N_trot = 3
theta0 = 0.1
eps_max = 0.2

# random local fields h_i ∈ [-1, 1]
rng = np.random.default_rng(42)
h = rng.uniform(-1.0, 1.0, size=n_qubits).tolist()

# initialize raw variational parameters to be zero)
raw_params = np.zeros(N_trot)
thetas = bounded_thetas(raw_params, theta0, eps_max).tolist()

# chaos sweep
lambdas = np.linspace(0.0, 1.0, 6)

for lam in lambdas:
    # ratio λ/θ (using mean θ)
    r = lam / np.mean(thetas)
    print(r)
    print(f"\nλ = {lam:.3f}, θ̄= {np.mean(thetas):.3f}, ratio r = {r:.2f}")

    # you could run the circuit, sample, or compute expectation values
    print(cudaq.draw(XXZ_model, float(lam), int(n_qubits), int(N_trot), list(thetas), list(h)))
