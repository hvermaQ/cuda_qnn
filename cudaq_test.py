import math, random, numpy as np, cudaq

# -----------------------------
# Parameters
# -----------------------------
n_features = 2
n_qubits = n_features + 1
depth = 1
lambda_angles = [0.3]
J_r, lambda_r = 1, 1
h = [1.0] * n_qubits
features = [0.2, 0.5]

# Random intervention angles
def generate_flat_intervention_angles(seed: int):
    rng = random.Random(seed)
    flat = []
    for _ in range(1, n_qubits):
        flat.extend([rng.uniform(0, 2*math.pi) for _ in range(3)])
    return flat

flat_angles = generate_flat_intervention_angles(42)

# -----------------------------
# XXZ kernel (your version)
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
    for i in range(1, n_qubits):
        ry(features[i-1], q[i])
    for t in range(N_trot):
        step_angle = lambda_angles[t]
        for _ in range(J_r):
            for i in range(n_qubits - 1):
                cx(q[i], q[i+1])
                rx(step_angle, q[i+1])
                cx(q[i], q[i+1])
                ry(math.pi/2, q[i]); ry(math.pi/2, q[i+1])
                cx(q[i], q[i+1]); rx(step_angle, q[i+1]); cx(q[i], q[i+1])
                ry(-math.pi/2, q[i]); ry(-math.pi/2, q[i+1])
                cx(q[i], q[i+1]); rz(step_angle, q[i+1]); cx(q[i], q[i+1])
        for _ in range(lambda_r):
            for i in range(n_qubits):
                rz(2.0 * h[i], q[i])
        for i in range(1, n_qubits):
            rx(intervention_angles[3*(i-1)], q[i])
            ry(intervention_angles[3*(i-1)+1], q[i])
            rz(intervention_angles[3*(i-1)+2], q[i])
    mx(q[0])
    mx(q[1])


# -----------------------------
# 1️⃣ Statevector (pre-measurement)
# -----------------------------

state = cudaq.get_state(XXZ_model, lambda_angles, n_qubits, depth, J_r, lambda_r,
                      h, features, flat_angles)
print(state)
# -----------------------------
# 2️⃣ Sampling (measurement collapse)
# -----------------------------
result = cudaq.sample(XXZ_model, lambda_angles, n_qubits, depth, J_r, lambda_r,
                      h, features, flat_angles, shots_count=1000, qpu_id='statevector')
print("Sampled measurement results (post-measurement collapse):")
print(result)