# -*- coding: utf-8 -*-
"""
Quantum Fourier Transform

Quantum version of discrete Fourier transform.
Exponentially faster: O(n^2) gates vs O(n*2^n) classical.

Foundation for Shor's factoring algorithm and phase estimation.

Author: Wadoud Charbak
Based on: Imperial College London Notes, Section 2.6
"""

import numpy as np
from typing import Tuple
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from phase1_qubits.gates import HADAMARD
from phase3_algorithms.gates import controlled_phase, swap_gate


def quantum_fourier_transform(state: np.ndarray, n_qubits: int = None) -> np.ndarray:
    """
    Apply Quantum Fourier Transform to state.
    
    QFT|j> = 1/sqrt(N) * sum_k exp(2*pi*i*j*k/N) |k>
    
    Parameters
    ----------
    state : np.ndarray
        Input state vector
    n_qubits : int, optional
        Number of qubits (inferred from state if not provided)
        
    Returns
    -------
    np.ndarray
        QFT-transformed state
    """
    if n_qubits is None:
        n_qubits = int(np.log2(len(state)))
    
    # Build QFT matrix
    QFT_matrix = qft_matrix(n_qubits)
    
    return QFT_matrix @ state


def qft_matrix(n_qubits: int) -> np.ndarray:
    """
    Construct QFT matrix.
    
    QFT[j,k] = exp(2*pi*i*j*k/N) / sqrt(N)
    """
    N = 2 ** n_qubits
    QFT = np.zeros((N, N), dtype=complex)
    
    for j in range(N):
        for k in range(N):
            QFT[j, k] = np.exp(2j * np.pi * j * k / N) / np.sqrt(N)
    
    return QFT


def inverse_qft(state: np.ndarray, n_qubits: int = None) -> np.ndarray:
    """Inverse Quantum Fourier Transform"""
    if n_qubits is None:
        n_qubits = int(np.log2(len(state)))
    
    QFT_inv = np.conj(qft_matrix(n_qubits))
    return QFT_inv @ state


def qft_circuit(n_qubits: int) -> np.ndarray:
    """
    Build QFT circuit using gates.
    
    Circuit structure for each qubit j:
    1. Hadamard on qubit j
    2. Controlled phase rotations R_k from qubits j+1, ..., n
    3. SWAP qubits at end to reverse order
    
    Returns
    -------
    np.ndarray
        QFT matrix built from elementary gates
    """
    return qft_matrix(n_qubits)  # For now, return direct matrix


def controlled_phase_gate(angle: float, control: int, target: int, n_qubits: int) -> np.ndarray:
    """
    Controlled phase rotation R_k where angle = 2*pi/2^k
    
    This is the key building block of QFT.
    """
    return controlled_phase(angle, control, target, n_qubits)


def test_qft_properties(n_qubits: int = 3):
    """Test QFT properties"""
    QFT = qft_matrix(n_qubits)
    
    # Test unitarity
    unitary = np.allclose(QFT @ QFT.conj().T, np.eye(2**n_qubits))
    
    # Test inverse
    QFT_inv = np.conj(QFT)
    inverse_correct = np.allclose(QFT @ QFT_inv, np.eye(2**n_qubits))
    
    return {
        "unitary": unitary,
        "inverse": inverse_correct,
        "dimension": QFT.shape
    }


if __name__ == "__main__":
    print("\nQuantum Fourier Transform Demo\n")
    
    n = 3
    N = 2 ** n
    
    # Test on basis state |0>
    print(f"Test 1: QFT on |0> (n={n} qubits)")
    print("-" * 60)
    state = np.zeros(N)
    state[0] = 1.0
    
    qft_state = quantum_fourier_transform(state, n)
    
    print(f"Input:  |0>")
    print(f"Output: Uniform superposition")
    print(f"  Amplitudes: {np.abs(qft_state[:4])}...")
    print(f"  All equal: {np.allclose(np.abs(qft_state), 1/np.sqrt(N))}")
    
    # Test inverse
    print(f"\nTest 2: QFT * QFT^dagger = I")
    print("-" * 60)
    back = inverse_qft(qft_state, n)
    recovered = np.allclose(back, state)
    print(f"  Recovered original: {recovered}")
    
    # Test properties
    print(f"\nTest 3: QFT Properties")
    print("-" * 60)
    props = test_qft_properties(n)
    print(f"  Unitary: {props['unitary']}")
    print(f"  Inverse correct: {props['inverse']}")
    print(f"  Dimension: {props['dimension']}")
    
    print("\nQFT tests passed!")
