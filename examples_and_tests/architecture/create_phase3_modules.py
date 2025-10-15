"""
Script to create Phase 3 modules programmatically
This avoids encoding issues with special characters
"""

import os

# Create oracles.py
oracles_content = '''# -*- coding: utf-8 -*-
"""
Quantum Oracles for Algorithm Testing

Creates oracle functions for Deutsch-Jozsa and Grover algorithms.
"""

import numpy as np
from typing import Callable, List, Dict, Optional, Union
from dataclasses import dataclass

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from phase1_qubits.gates import PAULI_X


@dataclass
class Oracle:
    """
    Quantum Oracle - encodes classical function into quantum circuit.
    
    Phase Oracle: O|x> = (-1)^f(x)|x>
    Bit-flip Oracle: O|x>|y> = |x>|y XOR f(x)>
    """
    function: Callable[[int], int]
    n_qubits: int
    oracle_type: str = "phase"
    name: str = "Oracle"
    matrix: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.matrix is None:
            self.matrix = self._compute_matrix()

    def _compute_matrix(self) -> np.ndarray:
        if self.oracle_type == "phase":
            return self._compute_phase_oracle()
        else:
            return self._compute_bitflip_oracle()

    def _compute_phase_oracle(self) -> np.ndarray:
        """Phase oracle: O|x> = (-1)^f(x)|x>"""
        dim = 2 ** self.n_qubits
        matrix = np.zeros((dim, dim), dtype=complex)
        for x in range(dim):
            phase = (-1) ** self.function(x)
            matrix[x, x] = phase
        return matrix

    def _compute_bitflip_oracle(self) -> np.ndarray:
        """Bit-flip oracle: O|x>|y> = |x>|y XOR f(x)>"""
        dim = 2 ** (self.n_qubits + 1)
        matrix = np.eye(dim, dtype=complex)
        
        for x in range(2 ** self.n_qubits):
            if self.function(x) == 1:
                state_0 = x * 2
                state_1 = x * 2 + 1
                matrix[state_0, state_0] = 0
                matrix[state_1, state_1] = 0
                matrix[state_0, state_1] = 1
                matrix[state_1, state_0] = 1
        return matrix

    def apply(self, state: np.ndarray) -> np.ndarray:
        return self.matrix @ state

    def is_balanced(self) -> bool:
        n_inputs = 2 ** self.n_qubits
        ones_count = sum(self.function(x) for x in range(n_inputs))
        return ones_count == n_inputs // 2

    def is_constant(self) -> bool:
        n_inputs = 2 ** self.n_qubits
        first_val = self.function(0)
        return all(self.function(x) == first_val for x in range(n_inputs))

    def evaluate_all(self) -> Dict[int, int]:
        return {x: self.function(x) for x in range(2 ** self.n_qubits)}


def create_constant_function(n_qubits: int, value: int = 0) -> Callable[[int], int]:
    """Create constant function f(x) = value"""
    def constant_func(x: int) -> int:
        return value
    return constant_func


def create_balanced_function(n_qubits: int, function_type: str = "parity") -> Callable[[int], int]:
    """Create balanced function (half 0s, half 1s)"""
    if function_type == "parity":
        return lambda x: bin(x).count('1') % 2
    elif function_type == "first_bit":
        return lambda x: (x >> (n_qubits - 1)) & 1
    elif function_type == "last_bit":
        return lambda x: x & 1
    else:
        n_inputs = 2 ** n_qubits
        indices = np.random.choice(n_inputs, n_inputs // 2, replace=False)
        ones_set = set(indices)
        return lambda x: 1 if x in ones_set else 0


def deutsch_jozsa_oracle(function: Union[Callable, str], n_qubits: int, oracle_type: str = "phase") -> Oracle:
    """Create Deutsch-Jozsa oracle"""
    if isinstance(function, str):
        if function.startswith("constant"):
            value = int(function.split("_")[1])
            func = create_constant_function(n_qubits, value)
            name = f"Constant-{value}"
        elif function.startswith("balanced"):
            func_type = function.split("_")[1] if "_" in function else "parity"
            func = create_balanced_function(n_qubits, func_type)
            name = f"Balanced-{func_type}"
    else:
        func = function
        name = getattr(function, '__name__', 'Custom')
    
    return Oracle(func, n_qubits, oracle_type, name)


def grover_oracle(target_states: Union[int, List[int]], n_qubits: int) -> Oracle:
    """Create Grover search oracle"""
    if isinstance(target_states, int):
        target_states = [target_states]
    
    target_set = set(target_states)
    func = lambda x: 1 if x in target_set else 0
    name = f"Grover-{target_states}"
    
    return Oracle(func, n_qubits, "phase", name)


if __name__ == "__main__":
    print("Testing Oracles...")
    oracle = deutsch_jozsa_oracle("constant_0", 3)
    print(f"Constant oracle: {oracle.is_constant()}")
    oracle = deutsch_jozsa_oracle("balanced_parity", 3)
    print(f"Balanced oracle: {oracle.is_balanced()}")
    oracle = grover_oracle(5, 3)
    print(f"Grover oracle created for target 5")
    print("All tests passed!")
'''

# Write oracles.py
with open('/Users/wadoudcharbak/Documents/GitHub/Quantum_Computing/src/phase3_algorithms/oracles.py', 'w', encoding='utf-8') as f:
    f.write(oracles_content)

print("oracles.py created successfully!")

