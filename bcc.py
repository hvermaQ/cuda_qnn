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
    n_samples=1000, n_features=2, n_informative=2,
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

# =======================
# Torch tensors, labels reshaped for consistency with torch notations
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# =======================
# VQNN setup
# VQC has 2 parms per qubit per layer (rx, ry, rz) + entangling cx gates
n_qubits = 2
depth = 5
n_params = 4 * n_qubits * depth
hamiltonian = cudaq.spin.z(0)  # measure the z of qubit 0 for label prediction

# Variational quantum kernel
# 2 inputs for 2 features
# thetas for variational angles
@cudaq.kernel
def vqc_kernel(x0: float, x1: float, thetas: list[float]):
    q = cudaq.qvector(n_qubits)
    #Encode data
    ry(x0, q[0])
    ry(x1, q[1])
    # Variational layers
    idx = 0
    for _ in range(depth):
        # Encode features with reuploading
        #ry(x0, q[0])
        #ry(x1, q[1])
        for qubit in range(n_qubits):
            rx(thetas[idx], q[qubit])
            idx += 1
            ry(thetas[idx], q[qubit])
            idx += 1
            rz(thetas[idx], q[qubit])
            idx += 1
            rx(thetas[idx], q[qubit])
            idx += 1
        cx(q[0], q[1])
        cx(q[1], q[0])

# =======================
# ensure differentiability by defining a custom autograd function
# since torch works with CNN, use their terminology
# instantiate torch_autograd for forward and backward passes
class QuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, thetas, x_batch):
        ctx.save_for_backward(thetas, x_batch)
        ctx.shift = np.pi / 2

        # Sequential evaluation of the batch (safe for GPU)
        outs = []
        for x in x_batch:
            result = cudaq.observe(vqc_kernel, hamiltonian,
                                   float(x[0].item()), float(x[1].item()),
                                   thetas.tolist())
            outs.append(result.expectation())
        return torch.tensor(outs, dtype=torch.float32, device=x_batch.device).unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output):
        thetas, x_batch = ctx.saved_tensors
        shift = ctx.shift
        grad = torch.zeros_like(thetas, device=thetas.device)

        # Sequential parameter-shift rule
        for i in range(len(thetas)):
            thetas_plus = thetas.clone()
            thetas_minus = thetas.clone()
            thetas_plus[i] += shift
            thetas_minus[i] -= shift

            exp_plus, exp_minus = [], []
            for x in x_batch:
                r_plus = cudaq.observe(vqc_kernel, hamiltonian,
                                        float(x[0].item()), float(x[1].item()),
                                        thetas_plus.tolist())
                r_minus = cudaq.observe(vqc_kernel, hamiltonian,
                                         float(x[0].item()), float(x[1].item()),
                                         thetas_minus.tolist())
                exp_plus.append(r_plus.expectation())
                exp_minus.append(r_minus.expectation())

            # Compute mean over batch and multiply by grad_output (chain rule)
            grad[i] = ((torch.tensor(exp_plus, device=thetas.device) -
                        torch.tensor(exp_minus, device=thetas.device)).mean() / 2) * grad_output.mean()

        return grad, None

# =======================
# promote the quantum layer to a NN module
# define forward pass using the autograd function
class QuantumLayer(nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.thetas = nn.Parameter(torch.randn(n_params))

    def forward(self, x_batch):
        return QuantumFunction.apply(self.thetas, x_batch)

# =======================
# use layers as in above to define outputs from a classifier
class VQCClassifier(nn.Module):
    def __init__(self, q_layer):
        super().__init__()
        self.q_layer = q_layer
        self.fc = nn.Sigmoid()  # postprocessing layer/ not NN layer: map ⟨Z⟩ → probability

    def forward(self, x):
        return self.fc(self.q_layer(x)) #apply sigmoid after processing by quantum layer
        #return self.q_layer(x)  # no postprocessing, only expectation value used directly

# =======================
# instantiating the NN layers
q_layer = QuantumLayer(n_params)
model = VQCClassifier(q_layer).to(device)

#accuracy function
def compute_accuracy(model_of_device, x_true, y_true):
    with torch.no_grad():
        # Forward pass to get probabilities
        probs = model_of_device(x_true)
        # Convert probabilities to discrete labels using 0.5 threshold
        predicted_labels = (probs >= 0.5).int()
        # Compare with true labels and compute accuracy
        correct = (predicted_labels == y_true.int()).sum().item()
        accuracy = correct / len(y_true)
    return accuracy

# =======================
# running and training on the training set
#criterion = nn.MSELoss()  # MSE loss is sufficient
criterion = nn.BCELoss() # for probabilistic outputs
#optimizer = optim.SGD(params=model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.05)  # slightly lower lr for stability

epochs = 100
loss_trace = []
accu_trace = []
for epoch in range(epochs):
    optimizer.zero_grad()  # initialize gradients to zero
    outputs = model(x_train)  # use raw expectation values for training
    loss = criterion(outputs, y_train)  # loss using MSE
    loss.backward()  # backprop to compute gradients
    optimizer.step()  # update params

    acc = compute_accuracy(model, x_test, y_test)
    loss_trace.append(loss.item())
    accu_trace.append(acc)
    print(f"Epoch {epoch}, Loss = {loss.item():.4f}")
    print(f"Epoch {epoch}, Accuracy = {acc:.4f}")

# =======================
# plot training loss
plt.plot(loss_trace)
plt.title("Training Loss")
plt.show()

# plot accuracy trace
plt.plot(accu_trace)
plt.title("Accuracy during training")
plt.show()

# =======================
# evaluate accuracy of the model on test set
with torch.no_grad():
    # Forward pass to get probabilities
    probs = model(x_test)
    # Convert probabilities to discrete labels using 0.5 threshold
    predicted_labels = (probs >= 0.5).int()
    # Compare with true labels and compute accuracy
    correct = (predicted_labels == y_test.int()).sum().item()
    accuracy = correct / len(y_test)
print("Test Accuracy =", accuracy)
