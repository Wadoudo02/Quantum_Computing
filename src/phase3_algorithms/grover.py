# -*- coding: utf-8 -*-
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
    print("\nGrover's Algorithm Demo\n")
    
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
    
    print(f"\nMeasurement results (1000 shots):")
    for state_val in sorted(counts.keys()):
        prob = counts[state_val] / 1000
        bar = "#" * int(prob * 50)
        print(f"  |{state_val}> : {bar} {prob:.3f}")
    
    success_rate = counts[target] / 1000
    print(f"\nSuccess rate: {success_rate:.1%}")
    print("\nGrover's algorithm test passed!")
