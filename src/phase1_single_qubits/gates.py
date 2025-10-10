"""
Quantum Gates Implementation
============================

Single-qubit quantum gates as unitary operators.

Gates are represented as 2×2 complex matrices that act on qubit states.
All gates preserve normalisation (they're unitary: U†U = I).

Reference: Imperial Notes Section 1.1.2 and 2.2
"""

import numpy as np
from .qubit import Qubit


# ============================================================================
# Pauli Gates (Section 1.1.1, Equation 5)
# ============================================================================

# Pauli X (NOT gate) - bit flip
PAULI_X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

# Pauli Y - bit and phase flip
PAULI_Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

# Pauli Z - phase flip
PAULI_Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

# Identity
IDENTITY = np.array([
    [1, 0],
    [0, 1]
], dtype=complex)


# ============================================================================
# Hadamard Gate (Section 2.2, Equation 33)
# ============================================================================

HADAMARD = (1 / np.sqrt(2)) * np.array([
    [1, 1],
    [1, -1]
], dtype=complex)


# ============================================================================
# Phase Gates
# ============================================================================

# S gate (Phase gate) - quarter turn around Z axis
S_GATE = np.array([
    [1, 0],
    [0, 1j]
], dtype=complex)

# T gate (π/8 gate)
T_GATE = np.array([
    [1, 0],
    [0, np.exp(1j * np.pi / 4)]
], dtype=complex)


# ============================================================================
# Rotation Gates
# ============================================================================

def rotation_x(theta: float) -> np.ndarray:
    """
    Rotation around X axis of Bloch sphere.
    
    R_x(θ) = exp(-iθσ_x/2) = cos(θ/2)I - i·sin(θ/2)σ_x
    
    Parameters
    ----------
    theta : float
        Rotation angle in radians
    
    Returns
    -------
    np.ndarray
        2×2 unitary matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -1j * s],
        [-1j * s, c]
    ], dtype=complex)


def rotation_y(theta: float) -> np.ndarray:
    """
    Rotation around Y axis of Bloch sphere.
    
    R_y(θ) = exp(-iθσ_y/2) = cos(θ/2)I - i·sin(θ/2)σ_y
    
    Parameters
    ----------
    theta : float
        Rotation angle in radians
    
    Returns
    -------
    np.ndarray
        2×2 unitary matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -s],
        [s, c]
    ], dtype=complex)


def rotation_z(theta: float) -> np.ndarray:
    """
    Rotation around Z axis of Bloch sphere.
    
    R_z(θ) = exp(-iθσ_z/2)
    
    Parameters
    ----------
    theta : float
        Rotation angle in radians
    
    Returns
    -------
    np.ndarray
        2×2 unitary matrix
    """
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


def phase_shift(phi: float) -> np.ndarray:
    """
    General phase shift gate.
    
    P(φ) = [[1, 0], [0, e^(iφ)]]
    
    Parameters
    ----------
    phi : float
        Phase angle in radians
    
    Returns
    -------
    np.ndarray
        2×2 unitary matrix
    """
    return np.array([
        [1, 0],
        [0, np.exp(1j * phi)]
    ], dtype=complex)


# ============================================================================
# Gate Application Functions
# ============================================================================

def apply_gate(qubit: Qubit, gate: np.ndarray) -> Qubit:
    """
    Apply a quantum gate to a qubit.
    
    Mathematically: |ψ'⟩ = U|ψ⟩
    
    Parameters
    ----------
    qubit : Qubit
        Input quantum state
    gate : np.ndarray
        2×2 unitary matrix representing the gate
    
    Returns
    -------
    Qubit
        Transformed quantum state
    
    Examples
    --------
    >>> q = ket_0()
    >>> q_flipped = apply_gate(q, PAULI_X)
    >>> print(q_flipped)
    |1⟩
    """
    new_state = gate @ qubit.state
    return Qubit(new_state, normalize=False)


def x_gate(qubit: Qubit) -> Qubit:
    """
    Apply Pauli X gate (NOT gate).
    
    Effect: |0⟩ ↔ |1⟩ (bit flip)
    
    Bloch sphere: 180° rotation around X axis
    """
    return apply_gate(qubit, PAULI_X)


def y_gate(qubit: Qubit) -> Qubit:
    """
    Apply Pauli Y gate.
    
    Effect: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
    
    Bloch sphere: 180° rotation around Y axis
    """
    return apply_gate(qubit, PAULI_Y)


def z_gate(qubit: Qubit) -> Qubit:
    """
    Apply Pauli Z gate (phase flip).
    
    Effect: |0⟩ → |0⟩, |1⟩ → -|1⟩
    
    Bloch sphere: 180° rotation around Z axis
    """
    return apply_gate(qubit, PAULI_Z)


def hadamard(qubit: Qubit) -> Qubit:
    """
    Apply Hadamard gate.
    
    Effect: Creates superposition
    - |0⟩ → |+⟩ = (|0⟩ + |1⟩)/√2
    - |1⟩ → |−⟩ = (|0⟩ − |1⟩)/√2
    
    Bloch sphere: 180° rotation around axis between X and Z
    """
    return apply_gate(qubit, HADAMARD)


def s_gate(qubit: Qubit) -> Qubit:
    """
    Apply S gate (phase gate).
    
    Effect: |0⟩ → |0⟩, |1⟩ → i|1⟩
    
    Note: S² = Z (applying S twice gives Z gate)
    """
    return apply_gate(qubit, S_GATE)


def t_gate(qubit: Qubit) -> Qubit:
    """
    Apply T gate (π/8 gate).
    
    Effect: |0⟩ → |0⟩, |1⟩ → e^(iπ/4)|1⟩
    
    Note: T² = S (applying T twice gives S gate)
    """
    return apply_gate(qubit, T_GATE)


# ============================================================================
# Composite Operations
# ============================================================================

def apply_sequence(qubit: Qubit, gates: list[np.ndarray]) -> Qubit:
    """
    Apply a sequence of gates to a qubit.
    
    Gates are applied left-to-right: U_n ... U_2 U_1 |ψ⟩
    
    Parameters
    ----------
    qubit : Qubit
        Initial state
    gates : list of np.ndarray
        List of gate matrices to apply in order
    
    Returns
    -------
    Qubit
        Final state after all gates applied
    
    Examples
    --------
    >>> q = ket_0()
    >>> # Apply H then X
    >>> q_final = apply_sequence(q, [HADAMARD, PAULI_X])
    """
    current = qubit.copy()
    for gate in gates:
        current = apply_gate(current, gate)
    return current


def time_evolution(qubit: Qubit, hamiltonian: np.ndarray, time: float) -> Qubit:
    """
    Evolve a qubit under a Hamiltonian for time t.
    
    Uses: |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
    
    Parameters
    ----------
    qubit : Qubit
        Initial state |ψ(0)⟩
    hamiltonian : np.ndarray
        2×2 Hermitian matrix (energy operator)
    time : float
        Evolution time (ℏ = 1 units)
    
    Returns
    -------
    Qubit
        State after time evolution |ψ(t)⟩
    
    Examples
    --------
    >>> # Evolve under σ_z Hamiltonian
    >>> q = ket_plus()
    >>> H = PAULI_Z
    >>> q_evolved = time_evolution(q, H, np.pi/4)
    """
    # Calculate evolution operator U(t) = exp(-iHt)
    # Using matrix exponentiation
    evolution_operator = np.array(expm(-1j * hamiltonian * time))
    return apply_gate(qubit, evolution_operator)


# ============================================================================
# Utility Functions
# ============================================================================

def is_unitary(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if a matrix is unitary (U†U = I).
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to check
    tolerance : float
        Numerical tolerance for equality
    
    Returns
    -------
    bool
        True if matrix is unitary
    """
    if matrix.shape != (2, 2):
        return False
    
    product = matrix.conj().T @ matrix
    identity = np.eye(2)
    return np.allclose(product, identity, atol=tolerance)


def gate_to_bloch_rotation(gate: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Decompose a unitary gate into axis and angle of rotation.
    
    Any single-qubit unitary can be written as:
    U = exp(-iθ n·σ/2) where n is a unit vector
    
    Parameters
    ----------
    gate : np.ndarray
        2×2 unitary matrix
    
    Returns
    -------
    axis : np.ndarray
        Unit vector [n_x, n_y, n_z] defining rotation axis
    angle : float
        Rotation angle θ in radians
    """
    # Extract rotation angle from trace
    trace = np.trace(gate)
    angle = 2 * np.arccos(np.clip(np.abs(trace) / 2, -1, 1))
    
    if np.abs(angle) < 1e-10:
        # Identity (or global phase)
        return np.array([0, 0, 1]), 0
    
    # Extract rotation axis from Pauli decomposition
    # U = cos(θ/2)I - i·sin(θ/2)(n_x·σ_x + n_y·σ_y + n_z·σ_z)
    coeff = -1j / np.sin(angle / 2)
    
    n_x = coeff * np.trace(gate @ PAULI_X) / 2
    n_y = coeff * np.trace(gate @ PAULI_Y) / 2
    n_z = coeff * np.trace(gate @ PAULI_Z) / 2
    
    axis = np.array([n_x.real, n_y.real, n_z.real])
    axis = axis / np.linalg.norm(axis)  # Normalise
    
    return axis, angle


# Matrix exponential (scipy needed for full implementation)
def expm(matrix: np.ndarray) -> np.ndarray:
    """
    Matrix exponential using eigendecomposition.
    
    For 2×2 matrices only.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    diagonal = np.diag(np.exp(eigenvalues))
    return eigenvectors @ diagonal @ np.linalg.inv(eigenvectors)


# ============================================================================
# Gate Descriptions (for documentation/visualization)
# ============================================================================

GATE_INFO = {
    "X": {
        "name": "Pauli X (NOT)",
        "matrix": PAULI_X,
        "description": "Bit flip: |0⟩ ↔ |1⟩",
        "bloch": "180° rotation around X axis"
    },
    "Y": {
        "name": "Pauli Y",
        "matrix": PAULI_Y,
        "description": "Bit and phase flip",
        "bloch": "180° rotation around Y axis"
    },
    "Z": {
        "name": "Pauli Z",
        "matrix": PAULI_Z,
        "description": "Phase flip: |1⟩ → -|1⟩",
        "bloch": "180° rotation around Z axis"
    },
    "H": {
        "name": "Hadamard",
        "matrix": HADAMARD,
        "description": "Creates superposition",
        "bloch": "180° rotation around X+Z axis"
    },
    "S": {
        "name": "S (Phase)",
        "matrix": S_GATE,
        "description": "90° phase shift: |1⟩ → i|1⟩",
        "bloch": "90° rotation around Z axis"
    },
    "T": {
        "name": "T (π/8)",
        "matrix": T_GATE,
        "description": "45° phase shift",
        "bloch": "45° rotation around Z axis"
    }
}