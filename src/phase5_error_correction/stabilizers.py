"""
Stabilizer Formalism for Quantum Error Correction

This module implements the stabilizer formalism, a powerful mathematical framework
for understanding and designing quantum error correction codes.

The stabilizer formalism is based on the Pauli group and provides:
1. An efficient way to describe quantum states
2. A systematic method for syndrome measurement
3. A framework for analyzing error correction codes

Key Concepts:
- Pauli Group: All n-qubit Pauli operators {I, X, Y, Z}^⊗n with phases
- Stabilizer: Set of operators S such that S|ψ⟩ = |ψ⟩
- Stabilizer State: A state |ψ⟩ uniquely defined by its stabilizers
- Syndrome: Measurement outcomes that identify errors

Author: Quantum Computing Learning Project
Phase: 5 - Error Correction
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from itertools import product
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator, Pauli
from qiskit.quantum_info import pauli_basis, PauliList
from qiskit_aer import AerSimulator


class PauliOperator:
    """
    Representation of a Pauli operator with phase.

    A Pauli operator on n qubits: ±(I, X, Y, Z)^⊗n with possible i phase.

    Attributes:
        n_qubits (int): Number of qubits
        operator (str): Pauli string (e.g., 'IXYZ')
        phase (complex): Phase factor (±1, ±i)
    """

    def __init__(self, operator: str, phase: complex = 1.0):
        """
        Initialize a Pauli operator.

        Args:
            operator: String of Pauli operators (e.g., 'IXYZ')
            phase: Phase factor (default: 1.0)
        """
        self.operator = operator.upper()
        self.n_qubits = len(operator)
        self.phase = phase

        # Validate operator string
        valid_paulis = set('IXYZ')
        if not all(c in valid_paulis for c in self.operator):
            raise ValueError(f"Invalid Pauli operator: {operator}")

    def __mul__(self, other: 'PauliOperator') -> 'PauliOperator':
        """
        Multiply two Pauli operators.

        Uses the Pauli multiplication rules:
        - XX = YY = ZZ = I
        - XY = iZ, YZ = iX, ZX = iY
        - YX = -iZ, ZY = -iX, XZ = -iY

        Args:
            other: Another PauliOperator

        Returns:
            PauliOperator: Product of the two operators
        """
        if self.n_qubits != other.n_qubits:
            raise ValueError("Operators must have same number of qubits")

        result_phase = self.phase * other.phase
        result_operator = []

        # Pauli multiplication table
        mult_table = {
            ('I', 'I'): ('I', 1), ('I', 'X'): ('X', 1), ('I', 'Y'): ('Y', 1), ('I', 'Z'): ('Z', 1),
            ('X', 'I'): ('X', 1), ('X', 'X'): ('I', 1), ('X', 'Y'): ('Z', 1j), ('X', 'Z'): ('Y', -1j),
            ('Y', 'I'): ('Y', 1), ('Y', 'X'): ('Z', -1j), ('Y', 'Y'): ('I', 1), ('Y', 'Z'): ('X', 1j),
            ('Z', 'I'): ('Z', 1), ('Z', 'X'): ('Y', 1j), ('Z', 'Y'): ('X', -1j), ('Z', 'Z'): ('I', 1),
        }

        for p1, p2 in zip(self.operator, other.operator):
            pauli, phase_factor = mult_table[(p1, p2)]
            result_operator.append(pauli)
            result_phase *= phase_factor

        return PauliOperator(''.join(result_operator), result_phase)

    def commutes_with(self, other: 'PauliOperator') -> bool:
        """
        Check if this operator commutes with another.

        Two Pauli operators commute if they have an even number of positions
        where both are non-identity and different.

        Args:
            other: Another PauliOperator

        Returns:
            bool: True if operators commute
        """
        if self.n_qubits != other.n_qubits:
            raise ValueError("Operators must have same number of qubits")

        anti_commute_count = 0

        for p1, p2 in zip(self.operator, other.operator):
            # Anti-commute if both are non-identity and different
            if p1 != 'I' and p2 != 'I' and p1 != p2:
                anti_commute_count += 1

        return anti_commute_count % 2 == 0

    def __str__(self) -> str:
        """String representation."""
        phase_str = ''
        if np.isclose(self.phase, 1):
            phase_str = '+'
        elif np.isclose(self.phase, -1):
            phase_str = '-'
        elif np.isclose(self.phase, 1j):
            phase_str = '+i'
        elif np.isclose(self.phase, -1j):
            phase_str = '-i'
        return f"{phase_str}{self.operator}"

    def __repr__(self) -> str:
        return self.__str__()

    def to_qiskit(self) -> Pauli:
        """Convert to Qiskit Pauli operator."""
        return Pauli(self.operator)


class StabilizerCode:
    """
    Base class for stabilizer quantum error correction codes.

    A stabilizer code is defined by a set of commuting Pauli operators (stabilizers)
    that stabilize the code space.

    Attributes:
        n_physical (int): Number of physical qubits
        n_logical (int): Number of logical qubits
        stabilizers (List[PauliOperator]): List of stabilizer generators
        logical_operators (Dict): Logical X and Z operators
    """

    def __init__(
        self,
        n_physical: int,
        stabilizers: List[PauliOperator],
        logical_x: Optional[List[PauliOperator]] = None,
        logical_z: Optional[List[PauliOperator]] = None
    ):
        """
        Initialize a stabilizer code.

        Args:
            n_physical: Number of physical qubits
            stabilizers: List of stabilizer generators
            logical_x: Logical X operators (optional)
            logical_z: Logical Z operators (optional)
        """
        self.n_physical = n_physical
        self.stabilizers = stabilizers
        self.n_stabilizers = len(stabilizers)
        self.n_logical = n_physical - self.n_stabilizers

        # Validate stabilizers commute
        for i, s1 in enumerate(stabilizers):
            for s2 in stabilizers[i+1:]:
                if not s1.commutes_with(s2):
                    raise ValueError(f"Stabilizers {s1} and {s2} do not commute")

        self.logical_x = logical_x or []
        self.logical_z = logical_z or []

    def measure_syndrome(self, error: PauliOperator) -> List[int]:
        """
        Measure the syndrome of an error.

        The syndrome is the list of eigenvalues (±1) when measuring each stabilizer
        after the error has occurred.

        Args:
            error: The error that occurred

        Returns:
            List[int]: Syndrome (0 for +1 eigenvalue, 1 for -1 eigenvalue)
        """
        syndrome = []

        for stabilizer in self.stabilizers:
            # Check if error anti-commutes with stabilizer
            commutes = error.commutes_with(stabilizer)
            syndrome.append(0 if commutes else 1)

        return syndrome

    def get_syndrome_string(self, syndrome: List[int]) -> str:
        """Convert syndrome list to string."""
        return ''.join(map(str, syndrome))

    def __str__(self) -> str:
        """String representation."""
        s = f"Stabilizer Code [[{self.n_physical}, {self.n_logical}]]\n"
        s += f"Stabilizer Generators:\n"
        for i, stab in enumerate(self.stabilizers):
            s += f"  S{i+1}: {stab}\n"
        return s


class BitFlipStabilizerCode(StabilizerCode):
    """
    3-qubit bit-flip code in stabilizer formalism.

    Stabilizers: Z₀Z₁, Z₁Z₂
    Logical operators: X̄ = X₀X₁X₂, Z̄ = Z₀

    Code space: {|000⟩, |111⟩}
    """

    def __init__(self):
        """Initialize the 3-qubit bit-flip stabilizer code."""
        # Define stabilizers
        s1 = PauliOperator('ZZI', phase=1)  # Z₀Z₁
        s2 = PauliOperator('IZZ', phase=1)  # Z₁Z₂

        # Logical operators
        logical_x = [PauliOperator('XXX', phase=1)]  # X̄ = X₀X₁X₂
        logical_z = [PauliOperator('ZII', phase=1)]  # Z̄ = Z₀

        super().__init__(
            n_physical=3,
            stabilizers=[s1, s2],
            logical_x=logical_x,
            logical_z=logical_z
        )

    def decode_syndrome(self, syndrome: List[int]) -> str:
        """
        Decode syndrome to determine error location.

        Args:
            syndrome: Syndrome measurement [s1, s2]

        Returns:
            str: Description of error
        """
        syndrome_str = ''.join(map(str, syndrome))
        lookup = {
            '00': 'No error',
            '01': 'X error on qubit 2',
            '10': 'X error on qubit 0',
            '11': 'X error on qubit 1'
        }
        return lookup.get(syndrome_str, 'Unknown error')


class ShorStabilizerCode(StabilizerCode):
    """
    Shor's 9-qubit code in stabilizer formalism.

    This code has 8 stabilizer generators that protect against
    arbitrary single-qubit errors.

    The stabilizers are:
    - Bit-flip stabilizers (6 generators, 2 per block)
    - Phase-flip stabilizers (2 generators between blocks)
    """

    def __init__(self):
        """Initialize Shor's 9-qubit stabilizer code."""
        # Bit-flip stabilizers for each block
        # Block 1
        s1 = PauliOperator('ZZIIIIII I', phase=1)  # Z₀Z₁
        s2 = PauliOperator('IZZIIIII I', phase=1)  # Z₁Z₂

        # Block 2
        s3 = PauliOperator('IIIZZIII I', phase=1)  # Z₃Z₄
        s4 = PauliOperator('IIIIZZII I', phase=1)  # Z₄Z₅

        # Block 3
        s5 = PauliOperator('IIIIIIZZ I', phase=1)  # Z₆Z₇
        s6 = PauliOperator('IIIIIIZZZ', phase=1)  # Z₇Z₈

        # Phase-flip stabilizers (in X basis)
        s7 = PauliOperator('XXXIIIIII IIIXXXIII', phase=1)  # X₀X₁X₂ X₃X₄X₅
        s8 = PauliOperator('IIIXXXIII IIIIIIXXX', phase=1)  # X₃X₄X₅ X₆X₇X₈

        # Logical operators
        logical_x = [PauliOperator('XXXXXXXXX', phase=1)]  # X̄ = X₀...X₈
        logical_z = [PauliOperator('ZZZIIIIII', phase=1)]  # Z̄ = Z₀Z₁Z₂

        super().__init__(
            n_physical=9,
            stabilizers=[s1, s2, s3, s4, s5, s6, s7, s8],
            logical_x=logical_x,
            logical_z=logical_z
        )


class FiveQubitCode(StabilizerCode):
    """
    The 5-qubit perfect code.

    This is the smallest code that can correct arbitrary single-qubit errors.
    It encodes 1 logical qubit into 5 physical qubits.

    Stabilizers:
    - M₁ = XZZXI
    - M₂ = IXZZX
    - M₃ = XIXZZ
    - M₄ = ZXIXZ

    Code parameters: [[5, 1, 3]] (5 physical, 1 logical, distance 3)
    """

    def __init__(self):
        """Initialize the 5-qubit perfect code."""
        # Define stabilizers
        s1 = PauliOperator('XZZXI', phase=1)
        s2 = PauliOperator('IXZZX', phase=1)
        s3 = PauliOperator('XIXZZ', phase=1)
        s4 = PauliOperator('ZXIXZ', phase=1)

        # Logical operators
        logical_x = [PauliOperator('XXXXX', phase=1)]
        logical_z = [PauliOperator('ZZZZZ', phase=1)]

        super().__init__(
            n_physical=5,
            stabilizers=[s1, s2, s3, s4],
            logical_x=logical_x,
            logical_z=logical_z
        )

    def create_encoding_circuit(self) -> QuantumCircuit:
        """
        Create encoding circuit for the 5-qubit code.

        Returns:
            QuantumCircuit: Encoding circuit
        """
        qc = QuantumCircuit(5, name='5QubitEncode')

        # Encoding sequence (derived from stabilizer constraints)
        # Start with |0000⟩ state (qubits 1-4 in |0⟩)

        # Create entanglement structure
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)

        # Apply additional gates to satisfy stabilizer constraints
        qc.cz(1, 2)
        qc.cz(2, 3)
        qc.cz(3, 4)
        qc.cz(4, 0)

        return qc


def analyze_stabilizer_code(code: StabilizerCode):
    """
    Analyze properties of a stabilizer code.

    Args:
        code: The stabilizer code to analyze
    """
    print(f"\n{'='*70}")
    print(f"STABILIZER CODE ANALYSIS")
    print(f"{'='*70}")

    print(f"\nCode Parameters: [[{code.n_physical}, {code.n_logical}]]")
    print(f"Physical qubits: {code.n_physical}")
    print(f"Logical qubits: {code.n_logical}")
    print(f"Number of stabilizers: {code.n_stabilizers}")

    print(f"\nStabilizer Generators:")
    for i, stab in enumerate(code.stabilizers):
        print(f"  S{i+1}: {stab}")

    # Check commutativity
    print(f"\nCommutativity Check:")
    all_commute = True
    for i, s1 in enumerate(code.stabilizers):
        for j, s2 in enumerate(code.stabilizers[i+1:], start=i+1):
            commutes = s1.commutes_with(s2)
            if not commutes:
                print(f"  S{i+1} and S{j+1}: Do NOT commute ❌")
                all_commute = False

    if all_commute:
        print(f"  ✓ All stabilizers commute")

    # Logical operators
    if code.logical_x:
        print(f"\nLogical X Operators:")
        for i, lx in enumerate(code.logical_x):
            print(f"  X̄{i}: {lx}")

    if code.logical_z:
        print(f"\nLogical Z Operators:")
        for i, lz in enumerate(code.logical_z):
            print(f"  Z̄{i}: {lz}")

    # Test some errors
    print(f"\nSyndrome Table (Single-Qubit X Errors):")
    print(f"{'Error':<20} {'Syndrome':<15} {'Detected?'}")
    print(f"{'-'*70}")

    for i in range(code.n_physical):
        error_str = 'I' * i + 'X' + 'I' * (code.n_physical - i - 1)
        error = PauliOperator(error_str)
        syndrome = code.measure_syndrome(error)
        syndrome_str = code.get_syndrome_string(syndrome)
        detected = any(syndrome)

        print(f"X{i:<19} {syndrome_str:<15} {'Yes ✓' if detected else 'No ✗'}")


def demonstrate_stabilizers():
    """Demonstrate the stabilizer formalism."""
    print("=" * 70)
    print("STABILIZER FORMALISM DEMONSTRATION")
    print("=" * 70)

    print("\n1. PAULI OPERATORS")
    print("-" * 70)

    # Create some Pauli operators
    x = PauliOperator('X')
    y = PauliOperator('Y')
    z = PauliOperator('Z')

    print(f"X: {x}")
    print(f"Y: {y}")
    print(f"Z: {z}")

    # Multiplication
    print(f"\nX * Y = {x * y}")
    print(f"Y * Z = {y * z}")
    print(f"Z * X = {z * x}")
    print(f"X * X = {x * x}")

    # Commutativity
    print(f"\nX commutes with Z? {x.commutes_with(z)}")
    print(f"X commutes with X? {x.commutes_with(x)}")

    print("\n2. 3-QUBIT BIT-FLIP CODE")
    print("-" * 70)

    bf_code = BitFlipStabilizerCode()
    analyze_stabilizer_code(bf_code)

    print("\n3. SHOR'S 9-QUBIT CODE")
    print("-" * 70)

    shor_code = ShorStabilizerCode()
    print(f"\nCode Parameters: [[{shor_code.n_physical}, {shor_code.n_logical}]]")
    print(f"Number of stabilizers: {shor_code.n_stabilizers}")
    print(f"\nStabilizers:")
    for i, stab in enumerate(shor_code.stabilizers):
        print(f"  S{i+1}: {stab}")

    print("\n4. 5-QUBIT PERFECT CODE")
    print("-" * 70)

    five_qubit_code = FiveQubitCode()
    analyze_stabilizer_code(five_qubit_code)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demonstrate_stabilizers()
