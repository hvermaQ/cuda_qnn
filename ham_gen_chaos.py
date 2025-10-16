#consider various chaotic hamiltonians to generate the circuit ansatz
#for now a single Hamiltonian with tunable chaos -- lambda

import cudaq
from cudaq import rx, ry, rz, cx
import math

import cudaq
from cudaq import rx, ry, rz, cx
import numpy as np
import math

# ==========================================================
# Helper: bounded reparameterization of variational angles
# ==========================================================
def bounded_thetas(raw_params, theta0=0.1, eps_max=0.2):
    """
    Maps unconstrained parameters p_k to bounded θ_k.
    θ_k ∈ [θ0(1 - eps_max), θ0(1 + eps_max)] via tanh transform.
    """
    return theta0 * (1.0 + eps_max * np.tanh(np.array(raw_params)))

# ==========================================================
# Kernel: integrable → chaotic → localized ansatz
# ==========================================================
@cudaq.kernel
def integrable_trotter_vqc(lambda_param: float,
                           n_qubits: int,
                           N_trot: int,
                           thetas: list[float],
                           h: list[float]):
    """
    thetas: one variational angle per Trotter step, shared across XX, YY, ZZ
    h: local fields (can be random)
    lambda_param: chaos control
    """
    q = cudaq.qvector(n_qubits)
    for k in range(N_trot):
        theta = thetas[k]  # shared angle for this Trotter step

        for i in range(n_qubits - 1):
            # XX coupling
            cx(q[i], q[i+1])
            rx(2 * theta, q[i+1])
            cx(q[i], q[i+1])

            # YY coupling
            ry(math.pi / 2, q[i])
            ry(math.pi / 2, q[i+1])
            cx(q[i], q[i+1])
            rx(2 * theta, q[i+1])
            cx(q[i], q[i+1])
            ry(-math.pi / 2, q[i])
            ry(-math.pi / 2, q[i+1])

            # ZZ coupling
            cx(q[i], q[i+1])
            rz(2 * theta, q[i+1])
            cx(q[i], q[i+1])

        # Local Z fields — break integrability, control chaos
        # no variational angle
        for i in range(n_qubits):
            rz(2 * lambda_param * h[i], q[i])

# ==========================================================
# Example usage / sweeping chaos ratio
# ==========================================================
if __name__ == "__main__":
    n_qubits = 4
    N_trot = 3
    theta0 = 0.1
    eps_max = 0.2

    # random local fields h_i ∈ [-1, 1]
    rng = np.random.default_rng(42)
    h = rng.uniform(-1.0, 1.0, size=n_qubits).tolist()

    # initialize raw variational parameters (unconstrained)
    raw_params = np.zeros(N_trot)
    thetas = bounded_thetas(raw_params, theta0, eps_max).tolist()

    # chaos sweep
    lambdas = np.linspace(0.0, 1.0, 6)

    for lam in lambdas:
        # ratio λ/θ (using mean θ)
        r = lam / np.mean(thetas)
        print(f"\nλ = {lam:.3f}, θ̄ = {np.mean(thetas):.3f}, ratio r = {r:.2f}")

        # you could run the circuit, sample, or compute expectation values
        kernel = integrable_trotter_vqc(lam, n_qubits, N_trot, thetas, h)
        # Example: just print or draw for inspection
        cudaq.draw(kernel)
