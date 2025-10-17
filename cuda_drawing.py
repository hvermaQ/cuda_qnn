import math, random, numpy as np, cudaq

# -----------------------------
# Parameters
# -----------------------------
n_qubits = 3  # q0 = bath, q1-q2 = system
depth = 1
lambda_angles = [0.3]
J_r, lambda_r = 1, 1
h = [1.0] * n_qubits
features = [0.2, 0.5]
shots = 1000

# Random interventions
def generate_flat_intervention_angles(seed: int):
    rng = random.Random(seed)
    return [rng.uniform(0, 2*math.pi) for _ in range(3*(n_qubits-1))]

flat_angles = generate_flat_intervention_angles(42)

# -----------------------------
# XXZ Kernel (measure system qubits only)
# -----------------------------
@cudaq.kernel
def XXZ_model(lambda_angles: list[float],
              n_qubits: int,
              N_trot: int,
              J_r: int,
              lambda_r: int,
              h: list[float],
              features: list[float],
              intervention_angles: list[float]):

    q = cudaq.qvector(n_qubits)

    # Encode features on system qubits (skip bath)
    for i in range(1, n_qubits):
        ry(features[i-1], q[i])

    for t in range(N_trot):
        step = lambda_angles[t]
        for i in range(n_qubits-1):
            cx(q[i], q[i+1])
            rx(step, q[i+1])
            cx(q[i], q[i+1])

        for i in range(n_qubits):
            rz(2.0 * h[i], q[i])

        for i in range(1, n_qubits):
            rx(intervention_angles[3*(i-1)], q[i])
            ry(intervention_angles[3*(i-1)+1], q[i])
            rz(intervention_angles[3*(i-1)+2], q[i])

    # Measure each system qubit individually
    mx(q[1])
    mx(q[2])


# -----------------------------
# 1️⃣ Get statevector
# -----------------------------
state_full = cudaq.get_state(XXZ_model, lambda_angles, n_qubits, depth,
                             J_r, lambda_r, h, features, flat_angles)
sv = np.array(state_full)
print("Statevector amplitudes (full 3-qubit state):", sv)

# -----------------------------
# 2️⃣ System-qubit probabilities (trace out bath)
# -----------------------------
sv = sv.reshape([2,2,2])  # q0, q1, q2
rho_system = np.sum(np.abs(sv)**2, axis=0).flatten()
print("\nSystem-qubit probabilities (q1,q2):")
for i, p in enumerate(rho_system):
    print(f"{i:02b}: {p:.3f}")

# -----------------------------
# 3️⃣ Sample system qubits
# -----------------------------
sample_counts = cudaq.sample(XXZ_model, lambda_angles, n_qubits, depth,
                             J_r, lambda_r, h, features, flat_angles,
                             shots_count=shots)
print("\nSampled counts (system qubits):")
for k, v in sample_counts.items():
    # Handle tuple or string keys
    if isinstance(k, tuple):
        key_str = ''.join(str(b) for b in k)
    else:
        key_str = str(k)
    print(f"{key_str}: {v}")

# -----------------------------
# 4️⃣ Compare exact vs sampled
# -----------------------------
print("\nComparison (exact vs sampled fraction):")
for k, v in sample_counts.items():
    # Map key to index for rho_system
    if isinstance(k, tuple):
        idx = int(''.join(str(b) for b in k), 2)
    else:
        idx = int(k, 2)
    exact_prob = rho_system[idx]
    sampled_frac = v / shots

    if isinstance(k, tuple):
        key_str = ''.join(str(b) for b in k)
    else:
        key_str = str(k)

    print(f"{key_str} → exact: {exact_prob:.3f}, sampled: {sampled_frac:.3f}")
