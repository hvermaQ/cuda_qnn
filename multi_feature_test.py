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

#hamiltonian or observle is a sum of z, can be tweaked
hamiltonian = cudaq.spin.z(0)  # measure the z of qubit 0 for label prediction
#for i in range(1, n_qubits):
#    hamiltonian = cudaq.spin.z(i)  # measure the z of qubit i for label prediction

# running routine on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudaq.set_target("nvidia")  # run all quantum circuits on NVIDIA GPU

# generate data for classification
x_full, y_full = make_classification(
    n_samples=100, n_features=n_qubits, n_informative=n_qubits,
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
plt.title("Synthetic Classification Dataset (%s features), one label")
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
              h: list[float],
              features: list[float]):
    """
    thetas: one variational angle per Trotter step, shared across XX, YY, ZZ
    h: local fields (can be random)
    lambda_param: chaos control
    """
    q = cudaq.qvector(n_qubits)

    #feature encoding
    for i in range(n_qubits):
        ry(features[i], q[i])

    for k in range(N_trot):
        theta = thetas[k]  # shared angle for this Trotter step

        for i in range(n_qubits - 1):
            # XX coupling
            cx(q[i], q[i+1])
            rx(2 * theta, q[i+1])
            cx(q[i], q[i+1])

            # YY coupling
            ry(math.pi/2, q[i])
            ry(math.pi/2, q[i+1])
            cx(q[i], q[i+1])
            rx(2 * theta, q[i+1])
            cx(q[i], q[i+1])
            ry(-math.pi/2, q[i])
            ry(-math.pi/2, q[i+1])

            # ZZ coupling
            cx(q[i], q[i+1])
            rz(2 * theta, q[i+1])
            cx(q[i], q[i+1])

        # Local Z fields — break integrability, control chaos
        # no variational angle
        for i in range(n_qubits):
            rz(2 * lambda_param * h[i], q[i])


# setting up the classifier and its appendages
# ensure differentiability by defining a custom autograd function
# since torch works with CNN, use their terminology
# instantiate torch_autograd for forward and backward passes
class QuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, thetas, x_batch, lambda_param, h):
        ctx.save_for_backward(thetas, x_batch)
        ctx.lambda_param = lambda_param
        ctx.h = h
        ctx.shift = np.pi / 2

        outs = []
        for x in x_batch:
            result = cudaq.observe(XXZ_model, hamiltonian,
                                   float(lambda_param), int(n_qubits),
                                   int(N_trot), thetas.tolist(), h, x.tolist())
            outs.append(result.expectation())
        return torch.tensor(outs, dtype=torch.float32, device=x_batch.device).unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output):
        thetas, x_batch = ctx.saved_tensors
        shift = ctx.shift
        lambda_param = ctx.lambda_param
        h = ctx.h
        grad = torch.zeros_like(thetas, device=thetas.device)

        for i in range(len(thetas)):
            thetas_plus = thetas.clone()
            thetas_minus = thetas.clone()
            thetas_plus[i] += shift
            thetas_minus[i] -= shift

            exp_plus, exp_minus = [], []
            for x in x_batch:
                r_plus = cudaq.observe(XXZ_model, hamiltonian,
                                       float(lambda_param), int(n_qubits),
                                       int(N_trot), thetas_plus.tolist(), h, x.tolist())
                r_minus = cudaq.observe(XXZ_model, hamiltonian,
                                        float(lambda_param), int(n_qubits),
                                        int(N_trot), thetas_minus.tolist(), h, x.tolist())
                exp_plus.append(r_plus.expectation())
                exp_minus.append(r_minus.expectation())

            grad[i] = ((torch.tensor(exp_plus, device=thetas.device) -
                        torch.tensor(exp_minus, device=thetas.device)).mean() / 2) * grad_output.mean()

        return grad, None, None, None

# promote the quantum layer to a NN module
# define forward pass using the autograd function
class QuantumLayer(nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.thetas = nn.Parameter(torch.randn(n_params))

    def forward(self, x_batch, lambda_param, h):
        return QuantumFunction.apply(self.thetas, x_batch, lambda_param, h)

# use layers as in above to define outputs from a classifier
class VQCClassifier(nn.Module):
    def __init__(self, q_layer):
        super().__init__()
        self.q_layer = q_layer
        self.fc = nn.Sigmoid()  # postprocessing layer/ not NN layer: map ⟨Z⟩ → probability

    def forward(self, x, lambda_param, h):
        return self.fc(self.q_layer(x, lambda_param, h)) #apply sigmoid after processing by quantum layer
        #return self.q_layer(x)  # no postprocessing, only expectation value used directly

#accuracy function
def compute_accuracy(model_of_device, x_true, y_true, lam, h):
    with torch.no_grad():
        probs = model_of_device(x_true, lam, h)
        predicted_labels = (probs >= 0.5).int()
        correct = (predicted_labels == y_true.int()).sum().item()
        accuracy = correct / len(y_true)
    return accuracy


#problem setup
N_trot = 8 #fixed depth
theta0 = 0.1 
eps_max = 0.2
n_params = N_trot
n_epochs = 100

# ansatz generating Hamiltonian parameter setup
rng = np.random.default_rng(42)
h = rng.uniform(-1.0, 1.0, size=n_qubits).tolist()

# Initialize variational angles (periodic)
raw_params = np.zeros(N_trot)
thetas = bounded_thetas(raw_params, theta0, eps_max).tolist()

# Coupling fixed as reference energy scale
J = 1.0

# Sweep over field strength λ → chaos control
lambdas = np.linspace(0.0, 2.0, 8)

final_losses, final_accuracies = [], []

for lam in lambdas:
    # Effective regime: λ small = integrable, λ ~ 1 = chaotic, λ large = localized
    print(f"\n=== λ = {lam:.3f} ===")

    # instantiating the NN layers
    q_layer = QuantumLayer(n_params)
    model = VQCClassifier(q_layer).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(torch.tensor(x_train[:, :n_qubits], dtype=torch.float32).to(device), lam, h)
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device))
        loss.backward()
        optimizer.step()

    # Evaluate accuracy
    acc = compute_accuracy(model,
                           torch.tensor(x_test[:, :n_qubits], dtype=torch.float32).to(device),
                           torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device),
                           lam, h)

    final_losses.append(loss.item())
    final_accuracies.append(acc)

    print(f"Final Loss: {loss.item():.4f}, Accuracy: {acc:.3f}")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(lambdas, final_accuracies, 'o-', color='orange')
plt.xlabel("λ (Local Field Strength)")
plt.ylabel("Test Accuracy")
plt.title("VQC Performance vs Chaos Strength λ")
plt.grid(True)
plt.show()


# Helper function: compute chaos metric from the circuit
def circuit_chaos_ratio(model, x_batch, lambda_param, h):
    """
    Compute a proxy for chaos directly from the quantum circuit.
    Here we use the variance of Z-expectation values across the batch.
    Higher variance indicates more chaotic / scrambled behavior.
    """
    with torch.no_grad():
        outputs = model(x_batch, lambda_param, h)
    # variance across batch
    ratio = torch.var(outputs).item()
    return ratio

# Target chaos ratio we want to maintain
target_ratio = 0.05  # choose based on observed variance
lambda_penalty_weight = 10.0  # weight for penalty from deviated chaos

# Sweep over field strength λ → chaos control
final_losses, final_accuracies = [], []

for lam in lambdas:
    # Effective regime: λ small = integrable, λ ~ 1 = chaotic, λ large = localized
    print(f"\n=== λ = {lam:.3f} ===")

    # instantiating the NN layers
    q_layer = QuantumLayer(n_params)
    model = VQCClassifier(q_layer).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Prepare input batch
        x_batch = torch.tensor(x_train[:, :n_qubits], dtype=torch.float32).to(device)
        y_batch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

        # Forward pass
        outputs = model(x_batch, lam, h)

        # Standard task loss
        task_loss = criterion(outputs, y_batch)

        # Compute chaos ratio directly from the quantum circuit
        current_ratio = circuit_chaos_ratio(model, x_batch, lam, h)

        # Penalty for deviating from target chaos ratio
        chaos_penalty = (current_ratio - target_ratio)**2

        # Total loss = task loss + weighted chaos penalty
        loss_total = task_loss + lambda_penalty_weight * chaos_penalty

        # Backward pass + optimization
        loss_total.backward()
        optimizer.step()

        # Optional: monitor progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Task Loss={task_loss.item():.4f}, "
                  f"Chaos Ratio={current_ratio:.4f}, Total Loss={loss_total.item():.4f}")

    # Evaluate accuracy
    acc = compute_accuracy(model,
                           torch.tensor(x_test[:, :n_qubits], dtype=torch.float32).to(device),
                           torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device),
                           lam, h)

    final_losses.append(loss_total.item())
    final_accuracies.append(acc)

    print(f"Final Loss: {loss_total.item():.4f}, Accuracy: {acc:.3f}, Chaos Ratio: {current_ratio:.4f}")

# Plot performance vs chaos strength λ
plt.figure(figsize=(8, 6))
plt.plot(lambdas, final_accuracies, 'o-', color='orange')
plt.xlabel("λ (Local Field Strength)")
plt.ylabel("Test Accuracy")
plt.title("VQC Performance vs Chaos Strength λ (with Chaos Penalty)")
plt.grid(True)
plt.show()
