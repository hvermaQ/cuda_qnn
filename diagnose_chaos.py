import math
import numpy as np
import cudaq
import matplotlib.pyplot as plt

# Parameters
n_qubits = 4
N_trot = 8
lambda_repeat_values = [1, 2, 4, 8, 16]  # only chaos control
n_samples = 20
perturbation = 1e-2
n_shots = 200
rng = np.random.default_rng(42)

# XXZ kernel with repeated λ subcircuit
@cudaq.kernel
def XXZ_model(lambda_angle: float,
              n_qubits: int,
              N_trot: int,
              lambda_repeat: int,
              h: list[float],
              features: list[float]):
    q = cudaq.qvector(n_qubits)
    for i in range(n_qubits):
        ry(features[i], q[i])
    for _ in range(N_trot):
        for i in range(n_qubits - 1):
            # XX
            cx(q[i], q[i+1])
            rx(lambda_angle, q[i+1])
            cx(q[i], q[i+1])
            # YY
            ry(math.pi/2, q[i])
            ry(math.pi/2, q[i+1])
            cx(q[i], q[i+1])
            rx(lambda_angle, q[i+1])
            cx(q[i], q[i+1])
            ry(-math.pi/2, q[i])
            ry(-math.pi/2, q[i+1])
            # ZZ
            cx(q[i], q[i+1])
            rz(lambda_angle, q[i+1])
            cx(q[i], q[i+1])
        for _ in range(lambda_repeat):
            for i in range(n_qubits):
                rz(2*h[i], q[i])

# Chaos metric 1: variance of <Z>
def compute_variance_metric(lambda_angle, h, lambda_repeat):
    results = []
    for _ in range(n_samples):
        features = rng.uniform(0, 2*math.pi, n_qubits).tolist()
        obs = cudaq.spin.z(0)
        result = cudaq.observe(XXZ_model, obs, float(lambda_angle), n_qubits, N_trot, lambda_repeat, h, features)
        results.append(result.expectation())
    return np.array(results).var()

# Chaos metric 2: Loschmidt fidelity
def compute_loschmidt_metric(lambda_angle, h, lambda_repeat):
    fidelities = []
    for _ in range(n_samples):
        features = rng.uniform(0, 2*math.pi, n_qubits)
        features_perturbed = features + rng.uniform(-perturbation, perturbation, n_qubits)
        obs = cudaq.spin.z(0)
        result_orig = cudaq.observe(XXZ_model, obs, float(lambda_angle), n_qubits, N_trot, lambda_repeat, h, features.tolist())
        result_pert = cudaq.observe(XXZ_model, obs, float(lambda_angle), n_qubits, N_trot, lambda_repeat, h, features_perturbed.tolist())
        fid = 1.0 - abs(result_orig.expectation() - result_pert.expectation()) / 2.0
        fidelities.append(fid)
    return np.mean(fidelities)

def compute_entropy_metric(lambda_angle, h, lambda_repeat, n_shots=200):
    entropies = []
    obs_qubits = list(range(n_qubits))  # measure all qubits
    for _ in range(n_samples):
        features = rng.uniform(0, 2*math.pi, n_qubits).tolist()

        # Sample the kernel
        sampler = cudaq.sample(XXZ_model,
                               n_shots,
                               lambda_angle,
                               n_qubits,
                               N_trot,
                               lambda_repeat,
                               h,
                               features)

        counts = sampler.counts
        qubit_probs = []
        for i in range(n_qubits):
            p0 = sum(v for k,v in counts.items() if k[n_qubits-1-i]=='0') / n_shots
            p1 = 1 - p0
            p0 = max(p0, 1e-12)
            p1 = max(p1, 1e-12)
            qubit_probs.append(-p0*np.log2(p0) - p1*np.log2(p1))
        entropies.append(np.mean(qubit_probs))
    return np.mean(entropies)

# Random local fields
h = rng.uniform(-1.0, 1.0, n_qubits).tolist()
lambda_angle = math.pi / 4  # fixed variational angle

# Sweep lambda_repeat
variance_metrics = []
loschmidt_metrics = []
entropy_metrics = []

for lam_r in lambda_repeat_values:
    var_z = compute_variance_metric(lambda_angle, h, lam_r)
    fidelity = compute_loschmidt_metric(lambda_angle, h, lam_r)
    entropy = compute_entropy_metric(lambda_angle, h, lam_r, n_shots)

    variance_metrics.append(var_z)
    loschmidt_metrics.append(fidelity)
    entropy_metrics.append(entropy)

    print(f"lambda_repeat={lam_r}, Var <Z>={var_z:.4f}, Fidelity={fidelity:.4f}, Entropy={entropy:.4f}")

# Plot each metric separately
plt.figure(figsize=(8,5))
plt.plot(lambda_repeat_values, variance_metrics, 'o-', color='red')
plt.xlabel('lambda_repeat')
plt.ylabel('Variance of <Z>')
plt.title('Variance metric vs lambda_repeat')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(lambda_repeat_values, loschmidt_metrics, 's-', color='blue')
plt.xlabel('lambda_repeat')
plt.ylabel('Loschmidt fidelity (lower = more chaos)')
plt.title('Loschmidt echo vs lambda_repeat')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(lambda_repeat_values, entropy_metrics, 'd-', color='green')
plt.xlabel('lambda_repeat')
plt.ylabel('Average single-qubit entropy')
plt.title('Entropic measure vs lambda_repeat')
plt.grid(True)
plt.show()
