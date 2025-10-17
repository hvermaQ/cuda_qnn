import math
import numpy as np
import cudaq
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
n_features = 10
n_qubits = n_features + 1  # system + bath
depth = 10
lambda_angles = np.random.rand(depth).tolist()
J_r, lambda_r = 1, 1
h = [1] * n_qubits

n_samples = 20
epsilon = 0.1

np.random.seed(42)
features_dataset = np.random.rand(n_samples, n_features)
perturbed_dataset = features_dataset + epsilon

# -----------------------------
# Precompute all chaotic angles
# -----------------------------
# Flat array of size: (n_qubits-1) * 3 * max_chaos_layers * depth
max_chaos_layers = 10
chaotic_angles = np.random.rand((n_qubits-1) * 3 * max_chaos_layers * depth) * 2 * math.pi
chaotic_angles = chaotic_angles.tolist()

# -----------------------------
# XXZ kernel
# -----------------------------
@cudaq.kernel
def XXZ_model(lambda_angles: list[float],
              n_qubits: int,
              N_trot: int,
              J_r: int,
              lambda_r: int,
              h: list[float],
              features: list[float],
              chaotic_angles: list[float],
              N_chaos: int):  # number of chaotic layers per Trotter step

    q = cudaq.qvector(n_qubits)

    # Encode features
    for i in range(1, n_qubits):
        ry(features[i-1], q[i])

    for t in range(N_trot):
        step_angle = lambda_angles[t]

        # --- Integrable XXZ + field ---
        for _ in range(J_r):
            for i in range(n_qubits-1):
                cx(q[i], q[i+1])
                rx(step_angle, q[i+1])
                cx(q[i], q[i+1])

        for _ in range(lambda_r):
            for i in range(n_qubits):
                rz(2.0 * h[i], q[i])

        # --- Chaotic layers ---
        for layer in range(N_chaos):
            for i in range(1, n_qubits):
                idx = 3*((i-1) + (n_qubits-1)*layer + (n_qubits-1)*N_chaos*t)
                rx(chaotic_angles[idx+0], q[i])
                ry(chaotic_angles[idx+1], q[i])
                rz(chaotic_angles[idx+2], q[i])

# -----------------------------
# Fidelity computation
# -----------------------------
def compute_fidelity(sv1, sv2):
    return np.abs(sv1.overlap(sv2))**2

# -----------------------------
# Sweep over chaotic layers
# -----------------------------
N_chaos_list = list(range(0, max_chaos_layers+1))
avg_fidelities = []

for N_chaos in N_chaos_list:
    fidelities = []
    for orig_features, pert_features in zip(features_dataset, perturbed_dataset):
        sv_orig = cudaq.get_state(
            XXZ_model,
            lambda_angles, n_qubits, depth, J_r, lambda_r,
            h, orig_features, chaotic_angles, N_chaos
        )
        sv_pert = cudaq.get_state(
            XXZ_model,
            lambda_angles, n_qubits, depth, J_r, lambda_r,
            h, pert_features, chaotic_angles, N_chaos
        )
        fidelities.append(compute_fidelity(sv_orig, sv_pert))
    avg_fidelities.append(np.mean(fidelities))

# -----------------------------
# Plot
# -----------------------------
plt.plot(N_chaos_list, avg_fidelities, marker='o')
plt.xlabel("Chaotic layers per Trotter step")
plt.ylabel("Average fidelity across samples")
plt.title("Fidelity decay vs controlled chaos")
plt.ylim(0,1)
plt.show()
