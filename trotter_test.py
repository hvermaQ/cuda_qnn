import math, random, numpy as np, cudaq
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
n_features = 10
n_qubits = n_features + 1  # 4 system + 1 bath
J_r, lambda_r = 1, 1
h = [1] * n_qubits

n_samples = 20
epsilon = 0.5

np.random.seed(42)
features_dataset = np.random.rand(n_samples, n_features)
perturbed_dataset = features_dataset + epsilon

# Random interventions
def generate_flat_intervention_angles(seed: int, n_qubits=n_qubits):
    rng = random.Random(seed)
    return [rng.uniform(0, 2*math.pi) for _ in range(3*(n_qubits-1))]

flat_angles = generate_flat_intervention_angles(42)

# -----------------------------
# XXZ Kernel
# -----------------------------
@cudaq.kernel
def XXZ_model(lambda_angles: list[float],
              n_qubits: int,
              N_trot: int,
              J_r: int,
              lambda_r: int,
              h: list[float],
              features: list[float],
              intervention_angles: list[float],
              N_chaos: int):

    q = cudaq.qvector(n_qubits)
    for i in range(1, n_qubits):
        ry(features[i-1], q[i])

    for t in range(N_trot):
        step_angle = lambda_angles[t % len(lambda_angles)]

        # Integrable XXZ + field
        for _ in range(J_r):
            for i in range(n_qubits-1):
                cx(q[i], q[i+1])
                rx(step_angle, q[i+1])
                cx(q[i], q[i+1])

        for _ in range(lambda_r):
            for i in range(n_qubits):
                rz(2.0 * h[i], q[i])

        # Chaotic interventions
        for _ in range(N_chaos):
            for i in range(1, n_qubits):
                rx(intervention_angles[3*(i-1)], q[i])

# -----------------------------
# Fidelity function
# -----------------------------
def compute_overlap_probability(sv1, sv2):
    return np.abs(sv1.overlap(sv2))**2

# -----------------------------
# Sweep depth and chaotic layers
# -----------------------------
depth_list = list(range(1, 16))         # 1 → 15 Trotter steps
N_chaos_list = list(range(0, 11))      # 0 → 10 chaotic layers per step

fidelity_map = np.zeros((len(depth_list), len(N_chaos_list)))

lambda_angles = np.random.rand(max(depth_list)).tolist()

for i, depth in enumerate(depth_list):
    for j, N_chaos in enumerate(N_chaos_list):
        fidelities = []

        for orig_features, pert_features in zip(features_dataset, perturbed_dataset):
            sv_orig = cudaq.get_state(XXZ_model, lambda_angles, n_qubits, depth,
                                      J_r, lambda_r, h, orig_features, flat_angles, N_chaos)
            sv_pert = cudaq.get_state(XXZ_model, lambda_angles, n_qubits, depth,
                                      J_r, lambda_r, h, pert_features, flat_angles, N_chaos)
            fidelities.append(compute_overlap_probability(sv_orig, sv_pert))

        fidelity_map[i, j] = np.mean(fidelities)

# -----------------------------
# Plot 2D heatmap
# -----------------------------
plt.figure(figsize=(10,6))
plt.imshow(fidelity_map, origin='lower', aspect='auto', 
           extent=[min(N_chaos_list), max(N_chaos_list), min(depth_list), max(depth_list)],
           vmin=0, vmax=1, cmap='viridis')
plt.colorbar(label="Average Fidelity")
plt.xlabel("Chaotic layers per Trotter step")
plt.ylabel("Trotter depth")
plt.title("Quantum Fidelity Map: Integrable → Chaotic")
plt.show()
