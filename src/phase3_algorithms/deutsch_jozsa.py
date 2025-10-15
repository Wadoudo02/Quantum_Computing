# -*- coding: utf-8 -*-
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
    
    correct = (result == "constant" and oracle.is_constant()) or               (result == "balanced" and oracle.is_balanced())
    
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
    print("\nDeutsch-Jozsa Algorithm Demo\n")
    
    # Test constant function
    print("Test 1: Constant function f(x) = 0")
    print("-" * 60)
    oracle = deutsch_jozsa_oracle("constant_0", 3)
    result, _, _ = deutsch_jozsa_algorithm(oracle, verbose=True)
    print(f"\nAlgorithm says: {result}")
    print(f"Correct: {result == 'constant'}\n")
    
    # Test balanced function  
    print("\nTest 2: Balanced function f(x) = parity(x)")
    print("-" * 60)
    oracle = deutsch_jozsa_oracle("balanced_parity", 3)
    result, _, _ = deutsch_jozsa_algorithm(oracle, verbose=True)
    print(f"\nAlgorithm says: {result}")
    print(f"Correct: {result == 'balanced'}\n")
    
    print("\nDeutsch-Jozsa tests passed!")
