# -----------------------------
# Imports
# -----------------------------
import math
import random
import numpy as np
import cudaq

# -----------------------------
# Simulation parameters
# -----------------------------
n_features = 20
n_qubits = n_features + 1  # 4 system qubits + 1 bath
depth = 15
#lambda_angles = [0.1] * depth
lambda_angles = np.random.rand(depth).tolist()
J_r, lambda_r = 1, 1
h = np.random.uniform(-1,1,size=n_qubits)

n_samples = 25
epsilon = 0.1 # perturbation for butterfly effect

# -----------------------------
# Generate dataset
# -----------------------------
np.random.seed(42)
features_dataset = np.random.rand(n_samples, n_features)
perturbed_dataset = features_dataset + epsilon

# -----------------------------
# Host-side random interventions (flattened)
# -----------------------------
def generate_flat_intervention_angles(seed: int):
    rng = random.Random(seed)
    flat_angles = []
    for _ in range(1, n_qubits):  # system qubits only
        flat_angles.extend([
            rng.uniform(0, 2*math.pi),  # rx
            rng.uniform(0, 2*math.pi),  # ry
            rng.uniform(0, 2*math.pi)   # rz
        ])
    return flat_angles

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
              intervention_angles: list[float]):  # flattened

    q = cudaq.qvector(n_qubits)

    # Encode features on system qubits (leave q[0] as bath)
    for i in range(1, n_qubits):
        ry(features[i-1], q[i])

    for t in range(N_trot):
        step_angle = lambda_angles[t]

        # --- J (exchange) subcircuit ---
        for _ in range(J_r):
            for i in range(n_qubits - 1):
                cx(q[i], q[i+1])
                rx(step_angle, q[i+1])
                cx(q[i], q[i+1])

                ry(math.pi/2, q[i])
                ry(math.pi/2, q[i+1])
                cx(q[i], q[i+1])
                rx(step_angle, q[i+1])
                cx(q[i], q[i+1])
                ry(-math.pi/2, q[i])
                ry(-math.pi/2, q[i+1])

                cx(q[i], q[i+1])
                rz(step_angle, q[i+1])
                cx(q[i], q[i+1])

        # --- λ (field) subcircuit ---
        for _ in range(lambda_r):
            for i in range(n_qubits):
                rz(2.0 * h[i], q[i])

        # --- apply host-generated random interventions ---
        for i in range(1, n_qubits):
            rx(intervention_angles[3*(i-1)+0], q[i])
            ry(intervention_angles[3*(i-1)+1], q[i])
            rz(intervention_angles[3*(i-1)+2], q[i])

    # Apply final measurements (if any)
    #mx(q[0])
    #mx(q[1])
    #my(q[2])
    #my(q[3])
    #my(q[4])

# -----------------------------
# Generate fixed interventions
# -----------------------------
flat_angles = generate_flat_intervention_angles(seed=42)

def compute_overlap_probability(initial_state: cudaq.State, final_state: cudaq.State):
    """Compute probability of the overlap with the initial state"""
    overlap = initial_state.overlap(final_state)
    return np.abs(overlap)**2


# Function to be used below to calculate the partial trace yielding a density matrix.

# -----------------------------
# Sample original and perturbed datasets
# -----------------------------
original_samples = []
perturbed_samples = []

for orig_features, pert_features in zip(features_dataset, perturbed_dataset):
    # Sample original
    sv_orig = cudaq.get_state(
        XXZ_model,
        lambda_angles,
        n_qubits,
        depth,
        J_r,
        lambda_r,
        h,
        orig_features,
        flat_angles
    )
    original_samples.append(sv_orig)

    # Sample perturbed
    sv_pert = cudaq.get_state(
        XXZ_model,
        lambda_angles,
        n_qubits,
        depth,
        J_r,
        lambda_r,
        h,
        pert_features,
        flat_angles
    )
    perturbed_samples.append(sv_pert)

# -----------------------------
# Compute fidelities
# -----------------------------
fidelities = []
for sv_orig, sv_pert in zip(original_samples, perturbed_samples):
    fidd = compute_overlap_probability(sv_orig, sv_pert)
    fidelities.append(fidd)

# -----------------------------
# Print or visualize
# -----------------------------

import matplotlib.pyplot as plt

fidelities = np.array(fidelities)  # ensure it's a numpy array

plt.plot(fidelities, marker='o')
plt.xlabel("Sample index")
plt.ylabel("Fidelity")
plt.title("Butterfly effect: Fidelity vs Sample")

# fix y-axis scale to show small deviations from 1
y_min = min(fidelities) - 0.001
y_max = max(fidelities) + 0.001
plt.ylim(y_min, y_max)

plt.show()

print("Fidelities between original and perturbed states:")
print(np.average(fidelities))