"""
Two-Qubit Quantum Gates
========================

Multi-qubit quantum gates that operate on 2 qubits simultaneously.
These gates create entanglement and are essential for quantum computation.

Reference: Imperial Notes Section 2.2 (Basic Gate Operations) and Section 2.4
"""

import numpy as np


# ============================================================================
# CNOT (Controlled-NOT) Gate - Section 2.2, Equation 35
# ============================================================================

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

"""
CNOT (Controlled-NOT) Gate
===========================

**What it does in plain English:**
The CNOT gate has two qubits: a "control" qubit and a "target" qubit.
- If the control qubit is |0⟩, do nothing to the target
- If the control qubit is |1⟩, flip the target (apply NOT/X gate)

**Example:**
|00⟩ → |00⟩  (control is 0, so target stays 0)
|01⟩ → |01⟩  (control is 0, so target stays 1)
|10⟩ → |11⟩  (control is 1, so target flips: 0→1)
|11⟩ → |10⟩  (control is 1, so target flips: 1→0)

**Why it's important:**
- Creates entanglement between qubits
- Universal gate for quantum computing (with single-qubit gates)
- Used in quantum algorithms like teleportation and error correction

**Matrix representation:**
Basis order: |00⟩, |01⟩, |10⟩, |11⟩

    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]]

**From your notes:**
- Equation 35: Û_c = |0⟩⟨0| ⊗ 𝟙 + |1⟩⟨1| ⊗ σ_x
- Acts like: control ⊗ (identity or X gate)
"""


# ============================================================================
# SWAP Gate - Section 2.4.4, Equation 62-63
# ============================================================================

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

"""
SWAP Gate
=========

**What it does in plain English:**
Exchanges the states of two qubits. Simple as that!

**Example:**
|00⟩ → |00⟩  (nothing to swap)
|01⟩ → |10⟩  (first qubit gets 1, second gets 0)
|10⟩ → |01⟩  (first qubit gets 0, second gets 1)
|11⟩ → |11⟩  (nothing to swap)

**Think of it as:**
If you have qubit A in state α and qubit B in state β:
|ψ⟩|φ⟩ → |φ⟩|ψ⟩

**Why it's useful:**
- Moving quantum information between qubits
- Routing in quantum circuits
- Can be decomposed into 3 CNOT gates

**Matrix representation:**
Basis order: |00⟩, |01⟩, |10⟩, |11⟩

    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]]

**From your notes:**
- Equation 62: SWAP|Ψ⟩|Φ⟩ = |Φ⟩|Ψ⟩
"""


# ============================================================================
# Controlled-Phase Gate - Section 2.2, Equation 34
# ============================================================================

def controlled_phase(phi: float) -> np.ndarray:
    """
    Controlled-phase gate with angle φ.

    **What it does:**
    - If control is |0⟩, do nothing
    - If control is |1⟩ and target is |1⟩, add phase e^(iφ)

    Parameters
    ----------
    phi : float
        Phase angle in radians

    Returns
    -------
    np.ndarray
        4×4 controlled-phase gate matrix

    Examples
    --------
    >>> # π phase shift (equivalent to CZ gate)
    >>> cz = controlled_phase(np.pi)
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * phi)]
    ], dtype=complex)


# Controlled-Z gate (special case of controlled-phase with φ = π)
CZ = controlled_phase(np.pi)

"""
Controlled-Z (CZ) Gate
======================

**What it does:**
- If both qubits are |1⟩, flip the sign (multiply by -1)
- Otherwise, do nothing

**Example:**
|00⟩ → |00⟩
|01⟩ → |01⟩
|10⟩ → |10⟩
|11⟩ → -|11⟩  (gets a minus sign!)

**Symmetric property:**
Unlike CNOT, CZ is symmetric - it doesn't matter which qubit is "control"
"""


# ============================================================================
# Controlled-Unitary Gate (General Form) - Section 2.4.1
# ============================================================================

def controlled_u(unitary: np.ndarray) -> np.ndarray:
    """
    General controlled-unitary gate.

    **What it does:**
    - If control is |0⟩, do nothing
    - If control is |1⟩, apply the unitary U to the target

    Parameters
    ----------
    unitary : np.ndarray
        2×2 unitary matrix to control

    Returns
    -------
    np.ndarray
        4×4 controlled-U gate matrix

    **From notes:** Equation 49: Û_cu = |0⟩⟨0| ⊗ 𝟙 + |1⟩⟨1| ⊗ U

    Examples
    --------
    >>> from .gates import PAULI_X, HADAMARD
    >>> # CNOT is a controlled-X
    >>> cnot = controlled_u(PAULI_X)
    >>> # Controlled-Hadamard
    >>> ch = controlled_u(HADAMARD)
    """
    if unitary.shape != (2, 2):
        raise ValueError("Unitary must be a 2×2 matrix")

    # Build 4×4 matrix: |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U
    identity = np.eye(2, dtype=complex)

    result = np.zeros((4, 4), dtype=complex)
    # Top-left 2×2 block: identity (when control is 0)
    result[0:2, 0:2] = identity
    # Bottom-right 2×2 block: unitary (when control is 1)
    result[2:4, 2:4] = unitary

    return result


# ============================================================================
# √SWAP Gate - Section 2.4.4, Equation 64
# ============================================================================

SQRT_SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0.5*(1+1j), 0.5*(1-1j), 0],
    [0, 0.5*(1-1j), 0.5*(1+1j), 0],
    [0, 0, 0, 1]
], dtype=complex)

"""
√SWAP Gate (Square Root of SWAP)
=================================

**What it does:**
If you apply it twice, you get a full SWAP.
√SWAP · √SWAP = SWAP

**Why it exists:**
- Intermediate step between doing nothing and full swap
- Used in some quantum algorithms
- Creates partial entanglement

**From notes:** Equation 64
"""


# ============================================================================
# Helper Functions
# ============================================================================

def apply_two_qubit_gate(state: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """
    Apply a two-qubit gate to a two-qubit state vector.

    Parameters
    ----------
    state : np.ndarray
        4-element state vector representing |ψ⟩
        Basis: [|00⟩, |01⟩, |10⟩, |11⟩]
    gate : np.ndarray
        4×4 unitary matrix

    Returns
    -------
    np.ndarray
        Transformed state

    Examples
    --------
    >>> # Create |10⟩ state
    >>> state = np.array([0, 0, 1, 0], dtype=complex)
    >>> # Apply CNOT (should flip to |11⟩)
    >>> result = apply_two_qubit_gate(state, CNOT)
    >>> print(result)  # [0, 0, 0, 1] = |11⟩
    """
    if state.shape != (4,):
        raise ValueError("State must be a 4-element vector")
    if gate.shape != (4, 4):
        raise ValueError("Gate must be a 4×4 matrix")

    return gate @ state


def apply_gate_to_system(system, gate: np.ndarray):
    """
    Apply a two-qubit gate to a TwoQubitSystem object.

    Parameters
    ----------
    system : TwoQubitSystem
        The two-qubit system
    gate : np.ndarray
        4×4 unitary matrix

    Returns
    -------
    TwoQubitSystem
        New system with transformed state

    Examples
    --------
    >>> from .multi_qubit import two_ket_10
    >>> sys = two_ket_10()
    >>> result = apply_gate_to_system(sys, CNOT)
    >>> print(result)  # |11⟩
    """
    # Import here to avoid circular dependency
    from .multi_qubit import TwoQubitSystem

    new_state = apply_two_qubit_gate(system.state, gate)
    return TwoQubitSystem(new_state, normalize=False)


def tensor_product(state1: np.ndarray, state2: np.ndarray) -> np.ndarray:
    """
    Compute tensor product of two single-qubit states.

    Creates a two-qubit state: |ψ⟩ ⊗ |φ⟩

    Parameters
    ----------
    state1 : np.ndarray
        First qubit state (2 elements)
    state2 : np.ndarray
        Second qubit state (2 elements)

    Returns
    -------
    np.ndarray
        Combined state (4 elements)

    **Math:**
    If |ψ⟩ = α|0⟩ + β|1⟩ and |φ⟩ = γ|0⟩ + δ|1⟩, then:
    |ψ⟩⊗|φ⟩ = αγ|00⟩ + αδ|01⟩ + βγ|10⟩ + βδ|11⟩

    Examples
    --------
    >>> # Create |01⟩ from |0⟩ and |1⟩
    >>> ket_0 = np.array([1, 0])
    >>> ket_1 = np.array([0, 1])
    >>> state_01 = tensor_product(ket_0, ket_1)
    >>> print(state_01)  # [0, 1, 0, 0]
    """
    return np.kron(state1, state2)


# ============================================================================
# Two-Qubit Basis States
# ============================================================================

# Computational basis states
KET_00 = np.array([1, 0, 0, 0], dtype=complex)
KET_01 = np.array([0, 1, 0, 0], dtype=complex)
KET_10 = np.array([0, 0, 1, 0], dtype=complex)
KET_11 = np.array([0, 0, 0, 1], dtype=complex)


# ============================================================================
# Gate Information Dictionary
# ============================================================================

TWO_QUBIT_GATE_INFO = {
    "CNOT": {
        "name": "Controlled-NOT",
        "matrix": CNOT,
        "description": "Flips target if control is |1⟩",
        "ref": "Section 2.2, Equation 35",
        "simple": "If first qubit is 1, flip second qubit"
    },
    "SWAP": {
        "name": "SWAP",
        "matrix": SWAP,
        "description": "Exchanges two qubits",
        "ref": "Section 2.4.4, Equation 62",
        "simple": "Swap the states of two qubits"
    },
    "CZ": {
        "name": "Controlled-Z",
        "matrix": CZ,
        "description": "Phase flip if both qubits are |1⟩",
        "ref": "Section 2.4.1, Equation 46",
        "simple": "Add minus sign to |11⟩ state"
    },
    "√SWAP": {
        "name": "Square Root of SWAP",
        "matrix": SQRT_SWAP,
        "description": "Apply twice to get SWAP",
        "ref": "Section 2.4.4, Equation 64",
        "simple": "Halfway between nothing and full swap"
    }
}


# ============================================================================
# Validation
# ============================================================================

def is_two_qubit_unitary(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if a 4×4 matrix is unitary.

    A matrix U is unitary if U†U = I (preserves quantum state normalization)
    """
    if matrix.shape != (4, 4):
        return False

    product = matrix.conj().T @ matrix
    identity = np.eye(4)
    return np.allclose(product, identity, atol=tolerance)


# Verify all gates are unitary
assert is_two_qubit_unitary(CNOT), "CNOT is not unitary!"
assert is_two_qubit_unitary(SWAP), "SWAP is not unitary!"
assert is_two_qubit_unitary(CZ), "CZ is not unitary!"
assert is_two_qubit_unitary(SQRT_SWAP), "√SWAP is not unitary!"
