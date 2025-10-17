# basic testing process tensors
# the model has system + bath
# intereventions are done on the system
# typically butterfly effect is diagnosed based on different interventions
# here the goal is to study the butterfly effect as a function of the input features
# insert random unitaries on the system qubits as interventions
# fix interventions for different input states
# system_ nqubits = nfeatures
# bath_nqubits = n_bath
# trace out the bath at the end of the circuit
# hamiltonian for classification is some scalar function of the state of the system
# possibly based on some entaglement distribution
# 
# 1. calculate the variance of the loss over different inputs features
# 2. also calcuate the variance of the loss over different interventions

import math
import random
import numpy as np
import cudaq
import matplotlib.pyplot as plt

# XXZ kernel with repeated λ subcircuit
# Effective regime: λ small = integrable, λ ~ 1 = chaotic, λ large = localized
# for r = λ / J, if r<<1 = integrable -> 1/r repetition of J subcircuit
# for r~1 = chaotic, same number of repetitions of J and λ subcircuits
# for r>>1 = localized, r repetitions of λ subcircuit

import cudaq, math

rng = random.Random(42)  # deterministic RNG for interventions

@cudaq.kernel
def XXZ_model(lambda_angles: list[float],  # list of lambda angles, one per Trotter step
              n_qubits: int,              # total qubits (bath + system)
              N_trot: int,                # trotterization depth
              J_r: int,                 # repetitions of J (exchange) subcircuit
              lambda_r: int,             # repetitions of lambda (field) subcircuit
              h: list[float],             # onsite field coefficients
              features: list[float]):     # input features for system qubits

    q = cudaq.qvector(n_qubits)

    # Encode input features on system qubits (leave q[0] as bath)
    for i in range(1, n_qubits):
        ry(features[i-1], q[i])

    # Trotterized evolution
    for t in range(N_trot):
        # Current angle for this Trotter step
        step_angle = lambda_angles[t]

        # --- J (exchange) subcircuit ---
        for _ in range(J_r):
            for i in range(n_qubits - 1):
                # XX term
                cx(q[i], q[i+1])
                rx(step_angle, q[i+1])
                cx(q[i], q[i+1])

                # YY term
                ry(math.pi/2, q[i])
                ry(math.pi/2, q[i+1])
                cx(q[i], q[i+1])
                rx(step_angle, q[i+1])
                cx(q[i], q[i+1])
                ry(-math.pi/2, q[i])
                ry(-math.pi/2, q[i+1])

                # ZZ term
                cx(q[i], q[i+1])
                rz(step_angle, q[i+1])
                cx(q[i], q[i+1])

        # --- λ (field / onsite) subcircuit ---
        for _ in range(lambda_r):
            for i in range(n_qubits):
                rz(2.0 * h[i], q[i])

        # --- random unitary interventions on system qubits ---
        for i in range(1, n_qubits):
            rand_angle_x = rng.uniform(0, 2*math.pi)
            rand_angle_y = rng.uniform(0, 2*math.pi)
            rand_angle_z = rng.uniform(0, 2*math.pi)
            rx(rand_angle_x, q[i])
            ry(rand_angle_y, q[i])
            rz(rand_angle_z, q[i])


#preprocess the number of relative trotter repetitions based on the ratio being explored
def get_trotter_repeats(ratio: float):
    if ratio < 1.0:
        J_repeat = int(1.0 / ratio)
        lambda_repeat = 1
    elif math.isclose(ratio, 1.0):
        J_repeat = 1
        lambda_repeat = 1
    else:
        J_repeat = 1
        lambda_repeat = int(ratio)
    return J_repeat, lambda_repeat

# Simulation parameters
n_features = 4
n_qubits = n_features + 1  # system + bath
depth = 5
n_samples = 50
epsilon = 1e-2  # fixed perturbation magnitude

# XXZ kernel parameters
lambda_angles = [0.3] * depth #fiducial angle for now
ratio = 1.0 #fiducial for chaos
J_r, lambda_r = 1, 1
h = [0.1] * n_qubits

# -----------------------------
# Generate input dataset
# -----------------------------
np.random.seed(42)
features_dataset = np.random.rand(n_samples, n_features)  # shape (n_samples, n_features)

# Apply fixed epsilon perturbation to create perturbed dataset
perturbed_dataset = features_dataset + epsilon

fidelities = []

# Fidelity calculation

def fidelity(state1, state2):
    return abs(np.vdot(state1, state2))**2

for orig_features, pert_features in zip(features_dataset, perturbed_dataset):
    # simulate original state
    sv_orig = cudaq.sample(
        XXZ_model,
        lambda_angles,
        n_qubits,
        depth,
        J_r,
        lambda_r,
        h,
        orig_features
    )
    # simulate perturbed state
    sv_pert = cudaq.sample(
        XXZ_model,
        lambda_angles,
        n_qubits,
        depth,
        J_r,
        lambda_r,
        h,
        pert_features
    )
    # fidelity as butterfly-effect metric
    F = fidelity(sv_orig, sv_pert)
    fidelities.append(F)

fidelities = np.array(fidelities)

print("Fidelity statistics:")
print("Mean fidelity:", fidelities.mean())
print("Variance of fidelity:", fidelities.var())
