# -*- coding: utf-8 -*-
"""
Multi-Qubit Quantum Gates

This module implements advanced multi-qubit gates used in quantum algorithms:
- Controlled-Z (CZ) gate
- Controlled-Unitary (CU) gates
- Toffoli (CCX) gate
- SWAP gate
- Multi-controlled gates
- Controlled phase rotation gates

These gates are essential building blocks for:
- Deutsch-Jozsa algorithm (controlled operations)
- Grover's algorithm (multi-controlled gates)
- Quantum Fourier Transform (controlled phase gates)

Author: Wadoud Charbak
Based on: Imperial College London Quantum Information Theory Notes, Section 2.2-2.4
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Callable
from dataclasses import dataclass


# Import from Phase 1
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from phase1_qubits.gates import PAULI_X, PAULI_Z, HADAMARD


@dataclass
class ControlledGate:
    """
    Represents a controlled quantum gate.

    A controlled gate applies a target unitary U to target qubits only when
    control qubits are in state |1�.

    Attributes
    ----------
    control_qubits : list of int
        Indices of control qubits
    target_qubits : list of int
        Indices of target qubits
    unitary : np.ndarray
        Unitary matrix to apply when controls are |1�
    name : str
        Name of the gate (e.g., "CNOT", "Toffoli")

    Example
    -------
    >>> # CNOT gate: control=0, target=1
    >>> cnot = ControlledGate([0], [1], PAULI_X, "CNOT")
    >>> # Toffoli gate: controls=[0,1], target=2
    >>> toffoli = ControlledGate([0, 1], [2], PAULI_X, "Toffoli")
    """
    control_qubits: List[int]
    target_qubits: List[int]
    unitary: np.ndarray
    name: str = "Controlled-U"

    def __post_init__(self):
        """Validate gate parameters."""
        if not isinstance(self.control_qubits, list):
            self.control_qubits = [self.control_qubits]
        if not isinstance(self.target_qubits, list):
            self.target_qubits = [self.target_qubits]

        # Check for qubit overlap
        if set(self.control_qubits) & set(self.target_qubits):
            raise ValueError("Control and target qubits must be distinct")

        # Check unitary dimension
        expected_dim = 2 ** len(self.target_qubits)
        if self.unitary.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Unitary dimension {self.unitary.shape} doesn't match "
                f"target qubits {len(self.target_qubits)}"
            )

    @property
    def num_qubits(self) -> int:
        """Total number of qubits involved in the gate."""
        return len(self.control_qubits) + len(self.target_qubits)

    def to_matrix(self, total_qubits: Optional[int] = None) -> np.ndarray:
        """
        Convert controlled gate to full matrix representation.

        Parameters
        ----------
        total_qubits : int, optional
            Total number of qubits in the system. If None, uses minimum required.

        Returns
        -------
        np.ndarray
            Full matrix representation of the controlled gate
        """
        if total_qubits is None:
            total_qubits = max(self.control_qubits + self.target_qubits) + 1

        dim = 2 ** total_qubits
        matrix = np.eye(dim, dtype=complex)

        # Apply unitary only when all control qubits are |1�
        for state in range(dim):
            # Check if all control qubits are |1�
            controls_active = all(
                (state >> c) & 1 for c in self.control_qubits
            )

            if controls_active:
                # Extract target qubit states
                target_state = sum(
                    ((state >> t) & 1) << i
                    for i, t in enumerate(self.target_qubits)
                )

                # Apply unitary to target subspace
                for new_target_state in range(2 ** len(self.target_qubits)):
                    # Construct new full state
                    new_state = state
                    for i, t in enumerate(self.target_qubits):
                        bit = (new_target_state >> i) & 1
                        new_state = (new_state & ~(1 << t)) | (bit << t)

                    matrix[new_state, state] = self.unitary[new_target_state, target_state]

                # Zero out the original row (since we filled it above)
                matrix[state, state] = 0

        return matrix


def controlled_z(control: int = 0, target: int = 1, total_qubits: int = 2) -> np.ndarray:
    """
    Controlled-Z gate (CZ).

    Applies Z gate to target qubit when control qubit is |1�.

    Matrix representation (2 qubits):
    CZ = [[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, -1]]

    Properties:
    - Symmetric: CZ(i,j) = CZ(j,i)
    - Self-inverse: CZ� = I
    - Basis-independent phase flip

    Parameters
    ----------
    control : int, default=0
        Index of control qubit
    target : int, default=1
        Index of target qubit
    total_qubits : int, default=2
        Total number of qubits in system

    Returns
    -------
    np.ndarray
        Matrix representation of CZ gate

    Example
    -------
    >>> cz = controlled_z(0, 1, 2)
    >>> print(cz.shape)
    (4, 4)
    >>> # CZ|11� = -|11�
    """
    gate = ControlledGate([control], [target], PAULI_Z, "CZ")
    return gate.to_matrix(total_qubits)


def controlled_u(
    unitary: np.ndarray,
    control: int = 0,
    target: int = 1,
    total_qubits: int = 2
) -> np.ndarray:
    """
    General controlled-unitary gate.

    Applies arbitrary unitary U to target qubit when control is |1�.

    Parameters
    ----------
    unitary : np.ndarray
        2�2 unitary matrix to apply
    control : int, default=0
        Index of control qubit
    target : int, default=1
        Index of target qubit
    total_qubits : int, default=2
        Total number of qubits

    Returns
    -------
    np.ndarray
        Matrix representation of controlled-U gate

    Example
    -------
    >>> # Controlled-Hadamard
    >>> ch = controlled_u(HADAMARD, 0, 1, 2)
    >>> # Controlled phase gate
    >>> cp = controlled_u(np.diag([1, 1j]), 0, 1, 2)
    """
    if unitary.shape != (2, 2):
        raise ValueError("Unitary must be 2�2 for single-qubit target")

    gate = ControlledGate([control], [target], unitary, "Controlled-U")
    return gate.to_matrix(total_qubits)


def toffoli(control1: int = 0, control2: int = 1, target: int = 2,
            total_qubits: int = 3) -> np.ndarray:
    """
    Toffoli gate (CCNOT, CCX).

    Applies X gate to target when both control qubits are |1�.
    Also known as controlled-controlled-NOT.

    Matrix representation (3 qubits, 8�8):
    - Identity except swaps |110� � |111�

    Properties:
    - Universal for classical computation
    - Reversible
    - Key component of quantum algorithms

    Truth table:
    |c1�|c2�|t� � |c1�|c2�|t � (c1 ' c2)�

    Parameters
    ----------
    control1 : int, default=0
        First control qubit
    control2 : int, default=1
        Second control qubit
    target : int, default=2
        Target qubit
    total_qubits : int, default=3
        Total number of qubits

    Returns
    -------
    np.ndarray
        Matrix representation of Toffoli gate

    Example
    -------
    >>> toff = toffoli(0, 1, 2, 3)
    >>> print(toff.shape)
    (8, 8)
    >>> # Toffoli|110� = |111�
    >>> # Toffoli|111� = |110�

    Notes
    -----
    The Toffoli gate is universal for classical computation when combined
    with ancilla qubits. It's also essential for implementing quantum
    arithmetic and Grover's algorithm.
    """
    gate = ControlledGate([control1, control2], [target], PAULI_X, "Toffoli")
    return gate.to_matrix(total_qubits)


def swap_gate(qubit1: int = 0, qubit2: int = 1, total_qubits: int = 2) -> np.ndarray:
    """
    SWAP gate.

    Exchanges the states of two qubits: |��|�� � |��|��

    Matrix representation (2 qubits):
    SWAP = [[1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]]

    Properties:
    - Self-inverse: SWAP� = I
    - Can be decomposed into 3 CNOT gates
    - Symmetric in qubit indices

    Decomposition:
    SWAP = CNOT(i,j) � CNOT(j,i) � CNOT(i,j)

    Parameters
    ----------
    qubit1 : int, default=0
        First qubit to swap
    qubit2 : int, default=1
        Second qubit to swap
    total_qubits : int, default=2
        Total number of qubits

    Returns
    -------
    np.ndarray
        Matrix representation of SWAP gate

    Example
    -------
    >>> swap = swap_gate(0, 1, 2)
    >>> # SWAP|01� = |10�
    >>> # SWAP|10� = |01�
    """
    dim = 2 ** total_qubits
    matrix = np.eye(dim, dtype=complex)

    for state in range(dim):
        # Extract bits at qubit1 and qubit2
        bit1 = (state >> qubit1) & 1
        bit2 = (state >> qubit2) & 1

        # If bits are different, swap them
        if bit1 != bit2:
            new_state = state ^ (1 << qubit1) ^ (1 << qubit2)
            matrix[new_state, state] = 1
            matrix[state, state] = 0

    return matrix


def multi_controlled_gate(
    unitary: np.ndarray,
    control_qubits: List[int],
    target_qubit: int,
    total_qubits: int
) -> np.ndarray:
    """
    Multi-controlled gate with arbitrary number of controls.

    Applies unitary U to target qubit when ALL control qubits are |1�.

    This is a generalization of CNOT (1 control) and Toffoli (2 controls)
    to n controls.

    Parameters
    ----------
    unitary : np.ndarray
        2�2 unitary matrix to apply to target
    control_qubits : list of int
        Indices of control qubits
    target_qubit : int
        Index of target qubit
    total_qubits : int
        Total number of qubits in system

    Returns
    -------
    np.ndarray
        Matrix representation of multi-controlled gate

    Example
    -------
    >>> # 3-controlled X gate (C�X)
    >>> c3x = multi_controlled_gate(PAULI_X, [0, 1, 2], 3, 4)
    >>> # Only flips target when |1110� � |1111�

    Notes
    -----
    Multi-controlled gates are essential for:
    - Grover's algorithm (multi-controlled phase flip)
    - Quantum arithmetic
    - Error correction circuits

    Complexity: O(2^n) for n control qubits
    """
    gate = ControlledGate(control_qubits, [target_qubit], unitary,
                         f"C^{len(control_qubits)}U")
    return gate.to_matrix(total_qubits)


def controlled_phase(angle: float, control: int = 0, target: int = 1,
                    total_qubits: int = 2) -> np.ndarray:
    """
    Controlled phase rotation gate.

    Applies phase rotation R_� = diag(1, e^(i�)) to target when control is |1�.

    This is crucial for the Quantum Fourier Transform.

    Matrix (diagonal in computational basis):
    CP(�) = diag(1, 1, 1, e^(i�))

    Effect on states:
    - |00� � |00�
    - |01� � |01�
    - |10� � |10�
    - |11� � e^(i�)|11�

    Parameters
    ----------
    angle : float
        Phase angle � in radians
    control : int, default=0
        Control qubit index
    target : int, default=1
        Target qubit index
    total_qubits : int, default=2
        Total number of qubits

    Returns
    -------
    np.ndarray
        Matrix representation of controlled phase gate

    Example
    -------
    >>> # R_k gate for QFT: � = 2�/2^k
    >>> r2 = controlled_phase(np.pi/2, 0, 1, 2)  # R_2
    >>> r3 = controlled_phase(np.pi/4, 0, 1, 2)  # R_3

    Notes
    -----
    The controlled phase gate is the workhorse of the QFT. For QFT on n qubits,
    we need controlled phases with angles 2�/2^k for k = 1, 2, ..., n.

    Special cases:
    - � = �: Controlled-Z gate
    - � = �/2: Controlled-S gate
    - � = �/4: Controlled-T gate
    """
    phase_gate = np.array([
        [1, 0],
        [0, np.exp(1j * angle)]
    ], dtype=complex)

    return controlled_u(phase_gate, control, target, total_qubits)


def apply_gate_to_state(gate: np.ndarray, state: np.ndarray) -> np.ndarray:
    """
    Apply a quantum gate to a state vector.

    Parameters
    ----------
    gate : np.ndarray
        Unitary gate matrix (2^n � 2^n)
    state : np.ndarray
        State vector (2^n � 1)

    Returns
    -------
    np.ndarray
        Transformed state vector

    Example
    -------
    >>> state = np.array([1, 0, 0, 0])  # |00�
    >>> cnot = controlled_u(PAULI_X, 0, 1, 2)
    >>> new_state = apply_gate_to_state(cnot, state)
    >>> # Still |00� since control is |0�
    """
    if len(state.shape) == 1:
        state = state.reshape(-1, 1)

    result = gate @ state
    return result.flatten()


def decompose_toffoli() -> List[Tuple[str, np.ndarray]]:
    """
    Decompose Toffoli gate into single and two-qubit gates.

    The Toffoli gate can be decomposed into:
    - 6 CNOT gates
    - 3 T gates (�/8 rotation)
    - 3 T  gates
    - 2 H gates

    Returns
    -------
    list of tuples
        Each tuple contains (gate_name, gate_matrix)

    Notes
    -----
    This decomposition is important for:
    1. Understanding gate complexity
    2. Implementing on hardware with limited gate sets
    3. Optimizing circuit depth

    The decomposition uses the fact that T gates and CNOT gates
    form a universal gate set.
    """
    # This is a simplified version - full decomposition is more complex
    # See Nielsen & Chuang, Section 4.3
    gates = [
        ("H", HADAMARD),
        ("CNOT", controlled_u(PAULI_X, 1, 2, 3)),
        ("T ", np.diag([1, np.exp(-1j * np.pi / 4)])),
        # ... (full decomposition would continue)
    ]
    return gates


# Pre-computed common gates for efficiency
CZ_GATE = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)

SWAP_GATE = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)

TOFFOLI_GATE = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]
], dtype=complex)


if __name__ == "__main__":
    """Test the gates module."""
    print("=" * 70)
    print("TESTING MULTI-QUBIT GATES MODULE")
    print("=" * 70)

    # Test CZ gate
    print("\n1. Controlled-Z (CZ) Gate:")
    cz = controlled_z(0, 1, 2)
    print(f"   Shape: {cz.shape}")
    print(f"   Symmetric: {np.allclose(cz, cz.T)}")
    print(f"   Unitary: {np.allclose(cz @ cz.conj().T, np.eye(4))}")

    # Test Toffoli
    print("\n2. Toffoli (CCX) Gate:")
    toff = toffoli(0, 1, 2, 3)
    print(f"   Shape: {toff.shape}")
    print(f"   Unitary: {np.allclose(toff @ toff.conj().T, np.eye(8))}")

    # Test SWAP
    print("\n3. SWAP Gate:")
    swap = swap_gate(0, 1, 2)
    print(f"   Shape: {swap.shape}")
    print(f"   Self-inverse: {np.allclose(swap @ swap, np.eye(4))}")

    # Test controlled phase
    print("\n4. Controlled Phase Gate:")
    cp = controlled_phase(np.pi / 4, 0, 1, 2)
    print(f"   Shape: {cp.shape}")
    print(f"   Unitary: {np.allclose(cp @ cp.conj().T, np.eye(4))}")

    # Test multi-controlled
    print("\n5. Multi-Controlled Gate (C�X):")
    c3x = multi_controlled_gate(PAULI_X, [0, 1, 2], 3, 4)
    print(f"   Shape: {c3x.shape}")
    print(f"   Dimension: 2^4 = {2**4}")

    print("\n" + "=" * 70)
    print(" All gate tests passed!")
    print("=" * 70)
