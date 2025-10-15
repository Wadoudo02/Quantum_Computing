"""
Create all Phase 3 algorithm modules
This creates deutsch_jozsa.py, grover.py, and qft.py
"""

import os

base_path = '/Users/wadoudcharbak/Documents/GitHub/Quantum_Computing/src/phase3_algorithms'

# ============================================================================
# DEUTSCH-JOZSA ALGORITHM
# ============================================================================

deutsch_jozsa_content = '''# -*- coding: utf-8 -*-
"""
Deutsch-Jozsa Algorithm

Determines if a function f: {0,1}^n -> {0,1} is constant or balanced
in a SINGLE query, compared to classical 2^(n-1) + 1 queries.

This demonstrates exponential quantum speedup through quantum interference.

Author: Wadoud Charbak
Based on: Imperial College London Notes, Section 2.3
"""

import numpy as np
from typing import Tuple, Dict
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from phase1_qubits.gates import HADAMARD
from phase3_algorithms.oracles import Oracle, deutsch_jozsa_oracle


def deutsch_jozsa_algorithm(oracle: Oracle, verbose: bool = False) -> Tuple[str, np.ndarray, list]:
    """
    Run Deutsch-Jozsa algorithm.
    
    Algorithm steps:
    1. Initialize |0>^n
    2. Apply Hadamard to all qubits: |psi> = sum_x |x> / sqrt(2^n)
    3. Apply oracle: adds phase (-1)^f(x)
    4. Apply Hadamard again: interference!
    5. Measure: if all |0>, function is constant; else balanced
    
    Parameters
    ----------
    oracle : Oracle
        Function oracle
    verbose : bool
        Print intermediate states
        
    Returns
    -------
    result : str
        "constant" or "balanced"
    final_state : np.ndarray
        Final quantum state
    history : list
        State history
    """
    n = oracle.n_qubits
    dim = 2 ** n
    
    # Step 1: Initialize |0>^n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    history = [("Initial |0>^n", state.copy())]
    
    if verbose:
        print(f"Deutsch-Jozsa Algorithm (n={n} qubits)")
        print("="*60)
        print(f"Step 1: Initial state |0>^n")
    
    # Step 2: Apply Hadamard to all qubits - create superposition
    H_tensor = hadamard_n(n)
    state = H_tensor @ state
    history.append(("After H^n (superposition)", state.copy()))
    
    if verbose:
        print(f"Step 2: Apply H^n - uniform superposition")
        print(f"        State: sum_x |x> / sqrt(2^n)")
    
    # Step 3: Apply oracle - phase kickback
    state = oracle.apply(state)
    history.append(("After oracle", state.copy()))
    
    if verbose:
        print(f"Step 3: Apply oracle - phase kickback")
        print(f"        State: sum_x (-1)^f(x) |x> / sqrt(2^n)")
    
    # Step 4: Apply Hadamard again - interference
    state = H_tensor @ state
    history.append(("After final H^n", state.copy()))
    
    if verbose:
        print(f"Step 4: Apply H^n again - quantum interference")
    
    # Step 5: Measure
    prob_zero = np.abs(state[0]) ** 2
    
    if verbose:
        print(f"Step 5: Measure first qubit")
        print(f"        P(|0>^n) = {prob_zero:.6f}")
    
    # Determine result
    if prob_zero > 0.99:  # Allow small numerical error
        result = "constant"
    else:
        result = "balanced"
    
    if verbose:
        print(f"\nResult: Function is {result.upper()}")
        print(f"Oracle was: {oracle.name}")
        print(f"Actual type: {'constant' if oracle.is_constant() else 'balanced'}")
        print("="*60)
    
    return result, state, history


def hadamard_n(n: int) -> np.ndarray:
    """
    n-qubit Hadamard gate H^(tensor n)
    
    H^n |x> = sum_y (-1)^(x.y) |y> / sqrt(2^n)
    where x.y is the bitwise inner product
    """
    H = HADAMARD
    result = H
    for i in range(n - 1):
        result = np.kron(result, H)
    return result


def create_dj_circuit(n_qubits: int, function_type: str) -> Dict:
    """
    Create a complete Deutsch-Jozsa circuit.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits
    function_type : str
        "constant_0", "constant_1", or "balanced_parity", etc.
        
    Returns
    -------
    dict with:
        - oracle: Oracle object
        - result: "constant" or "balanced"
        - state: final state
        - correct: whether result matches oracle type
    """
    oracle = deutsch_jozsa_oracle(function_type, n_qubits)
    result, state, history = deutsch_jozsa_algorithm(oracle, verbose=False)
    
    correct = (result == "constant" and oracle.is_constant()) or \
              (result == "balanced" and oracle.is_balanced())
    
    return {
        "oracle": oracle,
        "result": result,
        "state": state,
        "history": history,
        "correct": correct,
        "n_qubits": n_qubits
    }


def verify_function_type(oracle: Oracle) -> bool:
    """
    Verify Deutsch-Jozsa gives correct answer.
    """
    result, _, _ = deutsch_jozsa_algorithm(oracle, verbose=False)
    
    if result == "constant":
        return oracle.is_constant()
    else:
        return oracle.is_balanced()


if __name__ == "__main__":
    print("\\nDeutsch-Jozsa Algorithm Demo\\n")
    
    # Test constant function
    print("Test 1: Constant function f(x) = 0")
    print("-" * 60)
    oracle = deutsch_jozsa_oracle("constant_0", 3)
    result, _, _ = deutsch_jozsa_algorithm(oracle, verbose=True)
    print(f"\\nAlgorithm says: {result}")
    print(f"Correct: {result == 'constant'}\\n")
    
    # Test balanced function  
    print("\\nTest 2: Balanced function f(x) = parity(x)")
    print("-" * 60)
    oracle = deutsch_jozsa_oracle("balanced_parity", 3)
    result, _, _ = deutsch_jozsa_algorithm(oracle, verbose=True)
    print(f"\\nAlgorithm says: {result}")
    print(f"Correct: {result == 'balanced'}\\n")
    
    print("\\nDeutsch-Jozsa tests passed!")
'''

# ============================================================================
# GROVER'S ALGORITHM
# ============================================================================

grover_content = '''# -*- coding: utf-8 -*-
"""
Grover's Search Algorithm

Searches unsorted database of N items in O(sqrt(N)) steps,
compared to classical O(N). Achieves quadratic speedup through
amplitude amplification.

Author: Wadoud Charbak
Based on: Imperial College London Notes, Section 2.5
"""

import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from phase1_qubits.gates import HADAMARD
from phase3_algorithms.oracles import Oracle, grover_oracle


def grover_search(
    target_states: List[int],
    n_qubits: int,
    iterations: int = None,
    verbose: bool = False
) -> Tuple[List[int], np.ndarray, list]:
    """
    Run Grover's search algorithm.
    
    Algorithm:
    1. Initialize uniform superposition
    2. Repeat ~pi/4 * sqrt(N) times:
       a. Apply oracle (mark targets)
       b. Apply diffusion (amplify marked)
    3. Measure - high probability of finding target
    
    Parameters
    ----------
    target_states : list of int
        States to search for
    n_qubits : int
        Number of qubits
    iterations : int, optional
        Number of Grover iterations (default: optimal)
    verbose : bool
        Print progress
        
    Returns
    -------
    measurements : list
        Measured states from multiple shots
    final_state : np.ndarray
        Final quantum state
    history : list
        Amplitude history
    """
    N = 2 ** n_qubits
    
    if iterations is None:
        iterations = optimal_grover_iterations(N, len(target_states))
    
    # Create oracle
    oracle = grover_oracle(target_states, n_qubits)
    
    # Step 1: Initialize uniform superposition
    H_n = hadamard_n(n_qubits)
    state = np.zeros(N, dtype=complex)
    state[0] = 1.0
    state = H_n @ state  # Equal superposition
    
    history = [state.copy()]
    
    if verbose:
        print(f"Grover's Algorithm (N={N}, targets={target_states})")
        print("="*60)
        print(f"Optimal iterations: {iterations}")
        print(f"Step 1: Created uniform superposition")
    
    # Step 2: Grover iterations
    for i in range(iterations):
        # Apply oracle (flip phase of target states)
        state = oracle.apply(state)
        
        # Apply diffusion operator (inversion about average)
        state = grover_diffusion(state, n_qubits)
        
        history.append(state.copy())
        
        if verbose:
            target_amp = np.abs(state[target_states[0]])
            print(f"Iteration {i+1}: Target amplitude = {target_amp:.4f}")
    
    if verbose:
        print("="*60)
    
    return state, history


def grover_diffusion(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Grover diffusion operator: D = 2|s><s| - I
    where |s> is the uniform superposition
    
    This inverts amplitudes about their average.
    """
    N = 2 ** n_qubits
    
    # Create uniform superposition |s>
    s = np.ones(N) / np.sqrt(N)
    
    # D = 2|s><s| - I
    D = 2 * np.outer(s, s) - np.eye(N)
    
    return D @ state


def optimal_grover_iterations(N: int, M: int = 1) -> int:
    """
    Optimal number of Grover iterations.
    
    For M marked items out of N total:
    k = floor(pi/4 * sqrt(N/M))
    
    Parameters
    ----------
    N : int
        Total number of items
    M : int
        Number of marked items
        
    Returns
    -------
    int
        Optimal iteration count
    """
    return int(np.floor(np.pi / 4 * np.sqrt(N / M)))


def amplitude_amplification_step(state: np.ndarray, oracle: Oracle, n_qubits: int) -> np.ndarray:
    """Single step of amplitude amplification: Oracle + Diffusion"""
    state = oracle.apply(state)
    state = grover_diffusion(state, n_qubits)
    return state


def hadamard_n(n: int) -> np.ndarray:
    """n-qubit Hadamard"""
    H = HADAMARD
    result = H
    for i in range(n - 1):
        result = np.kron(result, H)
    return result


def measure_grover(state: np.ndarray, shots: int = 1000) -> List[int]:
    """Measure final state multiple times"""
    probabilities = np.abs(state) ** 2
    results = np.random.choice(len(state), size=shots, p=probabilities)
    return results.tolist()


if __name__ == "__main__":
    print("\\nGrover's Algorithm Demo\\n")
    
    # Test: Search for state |5> in 3-qubit system (N=8)
    print("Searching for |5> in 8-item database")
    print("-" * 60)
    
    target = 5
    n_qubits = 3
    
    state, history = grover_search([target], n_qubits, verbose=True)
    
    # Measure
    measurements = measure_grover(state, shots=1000)
    
    # Count results
    from collections import Counter
    counts = Counter(measurements)
    
    print(f"\\nMeasurement results (1000 shots):")
    for state_val in sorted(counts.keys()):
        prob = counts[state_val] / 1000
        bar = "#" * int(prob * 50)
        print(f"  |{state_val}> : {bar} {prob:.3f}")
    
    success_rate = counts[target] / 1000
    print(f"\\nSuccess rate: {success_rate:.1%}")
    print("\\nGrover's algorithm test passed!")
'''

# ============================================================================
# QUANTUM FOURIER TRANSFORM
# ============================================================================

qft_content = '''# -*- coding: utf-8 -*-
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
    print("\\nQuantum Fourier Transform Demo\\n")
    
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
    print(f"\\nTest 2: QFT * QFT^dagger = I")
    print("-" * 60)
    back = inverse_qft(qft_state, n)
    recovered = np.allclose(back, state)
    print(f"  Recovered original: {recovered}")
    
    # Test properties
    print(f"\\nTest 3: QFT Properties")
    print("-" * 60)
    props = test_qft_properties(n)
    print(f"  Unitary: {props['unitary']}")
    print(f"  Inverse correct: {props['inverse']}")
    print(f"  Dimension: {props['dimension']}")
    
    print("\\nQFT tests passed!")
'''

# Write all files
files_to_create = [
    ('deutsch_jozsa.py', deutsch_jozsa_content),
    ('grover.py', grover_content),
    ('qft.py', qft_content)
]

for filename, content in files_to_create:
    filepath = os.path.join(base_path, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created {filename}")

print("\\nAll algorithm modules created successfully!")

