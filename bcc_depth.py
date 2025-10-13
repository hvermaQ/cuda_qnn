import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cudaq
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#device setup for classical part
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device setup for quantum part
cudaq.set_target("nvidia")

# generate data for binary classification problem
x_full, y_full = make_classification(
    n_samples=100, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)
x_full = (x_full - np.mean(x_full, axis=0)) / np.std(x_full, axis=0)
x_train, x_test, y_train, y_test = train_test_split(
    x_full, y_full, test_size=0.2, random_state=42
)

# Convert labels from {0,1} to {-1,+1}
#y_train = 2 * y_train - 1
#y_test = 2 * y_test - 1

# Torch tensors, labels reshaped for consistency with torch notations
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)


# Plot the data for clarity
plt.figure(figsize=(8, 6))
plt.scatter(x_full[:, 0], x_full[:, 1], c=y_full, cmap='bwr', edgecolors='k')
plt.title("Synthetic Classification Dataset (2 features), one label")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# VQNN setup
# VQC has 2 params per qubit per layer (rx, ry, rz) + entangling cx gates
depths = [1, 2, 3, 4, 5]  # circuit layers to try
n_qubits = 2 #qubits, two for the two features here, replace by autoencoder
accuracy_results = []

# Variational quantum kernel
# 2 inputs for 2 features
# thetas for variational angles
@cudaq.kernel
def vqc_kernel(x0: float, x1: float, thetas: list[float], depth: int):
    q = cudaq.qvector(n_qubits)
    # Encode features
    ry(x0, q[0])
    ry(x1, q[1])
    idx = 0
    # Variational layers using depth argument
    for _ in range(depth):
        for qubit in range(n_qubits):
            rx(thetas[idx], q[qubit]); idx += 1
            ry(thetas[idx], q[qubit]); idx += 1
            rz(thetas[idx], q[qubit]); idx += 1
        cx(q[0], q[1])

#hamiltonian is sum of the expectation values of z on both qubits
#hamiltonian = cudaq.spin.z(0) + cudaq.spin.z(1)  
hamiltonian = cudaq.spin.z(0) #measure the z of qubit 0 for label prediction

# ensure differentiability by defining a custom autograd function
# since torch works with CNN, use their terminology
# instantiate torch_autograd for forward and backward passes
class QuantumFunction(torch.autograd.Function):
    #forward pass
    @staticmethod
    def forward(ctx, thetas, x_batch, depth):
        #ctx : relevant data for backward pass
        ctx.save_for_backward(thetas, x_batch)
        ctx.shift = np.pi / 2
        ctx.depth = depth

        # Sequential batch evaluation
        outs = torch.zeros((x_batch.shape[0], 1), device=x_batch.device)
        for idx, x in enumerate(x_batch):
            result = cudaq.observe(vqc_kernel, hamiltonian,
                                   float(x[0].item()), float(x[1].item()),
                                   thetas.tolist(), depth)
            outs[idx, 0] = result.expectation()
        return outs

    #backward pass
    @staticmethod
    def backward(ctx, grad_output):
        thetas, x_batch = ctx.saved_tensors
        shift = ctx.shift
        depth = ctx.depth
        grad = torch.zeros_like(thetas, device=thetas.device)

        # Parameter-shift rule
        for i in range(len(thetas)):
            #+-pi/2 shift for eavluating gardients at the forwards pass angles
            thetas_plus = thetas.clone() 
            thetas_minus = thetas.clone()
            thetas_plus[i] += shift
            thetas_minus[i] -= shift
            exp_plus = torch.zeros((x_batch.shape[0],), device=x_batch.device)
            exp_minus = torch.zeros((x_batch.shape[0],), device=x_batch.device)
            for idx, x in enumerate(x_batch):
                r_plus = cudaq.observe(vqc_kernel, cudaq.spin.z(0),
                                        float(x[0].item()), float(x[1].item()),
                                        thetas_plus.tolist(), depth)
                r_minus = cudaq.observe(vqc_kernel, cudaq.spin.z(0),
                                        float(x[0].item()), float(x[1].item()),
                                        thetas_minus.tolist(), depth)
                exp_plus[idx] = r_plus.expectation() #expectation value for positive shift
                exp_minus[idx] = r_minus.expectation() #expectation value for negative shift
            #take mean of the gradient for all datapoints at each angle
            # parameter-shift rule to compute d<Z>/dtheta
            grad[i] = ((exp_plus - exp_minus).mean() / 2) * grad_output.mean()
        return grad, None, None

# promote the quantum layer to a NN module
# define forward pass using the autograd function
class QuantumLayer(nn.Module):
    def __init__(self, n_params, depth):
        super().__init__()
        self.thetas = nn.Parameter(torch.randn(n_params) * 0.1)
        self.depth = depth

    def forward(self, x_batch):
        return QuantumFunction.apply(self.thetas, x_batch, self.depth)

# use layers as in above to define outputs from a classifier
class VQCClassifier(nn.Module):
    def __init__(self, q_layer):
        super().__init__()
        self.q_layer = q_layer
        # self.fc = nn.Sigmoid()  # postprocessing layer/ not NN layer: map ⟨Z⟩ → probability

    def forward(self, x):
        # return self.fc(self.q_layer(x)) #apply sigmoid after processing by quantum layer
        return self.q_layer(x)  # no postprocessing, only expectation value used directly

# Loop over circuit depths
epochs = 100 

for depth in depths:
    n_params = 3 * n_qubits * depth
    #instantiate the layers and task
    q_layer = QuantumLayer(n_params, depth)
    model = VQCClassifier(q_layer).to(device)
    #optimizer instantiation with hyperparameters
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)  # params will be set in loop
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        #init gradients to zero
        optimizer.zero_grad()
        #forwards pass through model
        outputs = model(x_train)
        #calculate loss
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate accuracy
    with torch.no_grad():
        preds = model(x_test)
        predicted_labels = torch.sign(preds)
        acc = (predicted_labels.eq(y_test).sum().item()) / len(y_test)
    accuracy_results.append(acc)
    print(f"Depth {depth}: Test Accuracy = {acc:.4f}")

# =======================
# Plot depth vs accuracy
plt.figure(figsize=(8, 5))
plt.plot(depths, accuracy_results, marker='o')
plt.xlabel("Circuit Depth")
plt.ylabel("Test Accuracy")
plt.title("Effect of Circuit Depth on VQC Accuracy")
plt.grid(True)
plt.show()