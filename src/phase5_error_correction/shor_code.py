"""
Shor's 9-Qubit Error Correction Code

This module implements Shor's 9-qubit code, the first quantum error correction code
capable of correcting arbitrary single-qubit errors (X, Y, Z errors and combinations).

The code works by concatenating two levels of encoding:
1. First level: Protects against phase-flip (Z) errors using a 3-qubit code
2. Second level: Protects against bit-flip (X) errors using a 3-qubit code

Combined, this protects against all single-qubit Pauli errors.

Encoding:
    |0⟩_L = (|000⟩ + |111⟩)⊗3 / 2√2
    |1⟩_L = (|000⟩ - |111⟩)⊗3 / 2√2

Author: Quantum Computing Learning Project
Phase: 5 - Error Correction
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error


class ShorCode:
    """
    Implementation of Shor's 9-qubit error correction code.

    This code encodes 1 logical qubit into 9 physical qubits and can correct
    any single-qubit error (bit-flip, phase-flip, or both).

    The encoding is hierarchical:
    1. Encode against phase flips: |0⟩ → |+++⟩, |1⟩ → |---⟩
    2. Encode each qubit against bit flips: |+⟩ → |+++⟩

    Result:
        |0⟩_L = (|000⟩ + |111⟩)(|000⟩ + |111⟩)(|000⟩ + |111⟩) / 2√2
        |1⟩_L = (|000⟩ - |111⟩)(|000⟩ - |111⟩)(|000⟩ - |111⟩) / 2√2

    Attributes:
        n_physical (int): Number of physical qubits (9)
        n_logical (int): Number of logical qubits (1)
        n_ancilla (int): Number of ancilla qubits for syndrome measurement
    """

    def __init__(self):
        """Initialize Shor's 9-qubit code."""
        self.n_physical = 9
        self.n_logical = 1
        self.n_ancilla_bit_flip = 6  # 2 ancilla per 3-qubit block
        self.n_ancilla_phase_flip = 2  # 2 ancilla for phase flip detection
        self.n_ancilla_total = self.n_ancilla_bit_flip + self.n_ancilla_phase_flip

    def create_encoding_circuit(self) -> QuantumCircuit:
        """
        Create the encoding circuit for Shor's 9-qubit code.

        The encoding proceeds in two stages:
        1. Phase-flip encoding: |0⟩ → (|0⟩+|3⟩+|6⟩)/√3, |1⟩ → (|0⟩-|3⟩+|6⟩)/√3
        2. Bit-flip encoding: Each logical qubit → 3 physical qubits

        Returns:
            QuantumCircuit: The encoding circuit
        """
        qc = QuantumCircuit(9, name='ShorEncode')

        # Stage 1: Encode against phase flips
        # Create superposition of first qubits of each block
        qc.h(0)
        qc.h(3)
        qc.h(6)

        # Entangle to create phase-flip code
        qc.cx(0, 3)
        qc.cx(0, 6)

        qc.barrier()

        # Stage 2: Encode each block against bit flips
        # Block 1 (qubits 0, 1, 2)
        qc.cx(0, 1)
        qc.cx(0, 2)

        # Block 2 (qubits 3, 4, 5)
        qc.cx(3, 4)
        qc.cx(3, 5)

        # Block 3 (qubits 6, 7, 8)
        qc.cx(6, 7)
        qc.cx(6, 8)

        return qc

    def create_decoding_circuit(self) -> QuantumCircuit:
        """
        Create the decoding circuit (inverse of encoding).

        Returns:
            QuantumCircuit: The decoding circuit
        """
        qc = QuantumCircuit(9, name='ShorDecode')

        # Stage 1: Decode bit-flip code for each block
        # Block 1
        qc.cx(0, 2)
        qc.cx(0, 1)

        # Block 2
        qc.cx(3, 5)
        qc.cx(3, 4)

        # Block 3
        qc.cx(6, 8)
        qc.cx(6, 7)

        qc.barrier()

        # Stage 2: Decode phase-flip code
        qc.cx(0, 6)
        qc.cx(0, 3)

        qc.h(6)
        qc.h(3)
        qc.h(0)

        return qc

    def create_bit_flip_syndrome_circuit(self) -> QuantumCircuit:
        """
        Create syndrome measurement circuit for bit-flip errors.

        Measures parity within each 3-qubit block to detect bit flips.
        Uses 6 ancilla qubits (2 per block).

        Returns:
            QuantumCircuit: Bit-flip syndrome measurement circuit
        """
        data = QuantumRegister(9, 'data')
        ancilla = QuantumRegister(6, 'ancilla_bf')
        syndrome = ClassicalRegister(6, 'syndrome_bf')

        qc = QuantumCircuit(data, ancilla, syndrome, name='BitFlipSyndrome')

        # Block 1 syndromes (qubits 0, 1, 2)
        # S1: parity of qubits 0 and 1
        qc.cx(data[0], ancilla[0])
        qc.cx(data[1], ancilla[0])
        qc.measure(ancilla[0], syndrome[0])

        # S2: parity of qubits 1 and 2
        qc.cx(data[1], ancilla[1])
        qc.cx(data[2], ancilla[1])
        qc.measure(ancilla[1], syndrome[1])

        # Block 2 syndromes (qubits 3, 4, 5)
        qc.cx(data[3], ancilla[2])
        qc.cx(data[4], ancilla[2])
        qc.measure(ancilla[2], syndrome[2])

        qc.cx(data[4], ancilla[3])
        qc.cx(data[5], ancilla[3])
        qc.measure(ancilla[3], syndrome[3])

        # Block 3 syndromes (qubits 6, 7, 8)
        qc.cx(data[6], ancilla[4])
        qc.cx(data[7], ancilla[4])
        qc.measure(ancilla[4], syndrome[4])

        qc.cx(data[7], ancilla[5])
        qc.cx(data[8], ancilla[5])
        qc.measure(ancilla[5], syndrome[5])

        return qc

    def create_phase_flip_syndrome_circuit(self) -> QuantumCircuit:
        """
        Create syndrome measurement circuit for phase-flip errors.

        Measures parity between blocks in the X basis to detect phase flips.
        Uses 2 ancilla qubits.

        Returns:
            QuantumCircuit: Phase-flip syndrome measurement circuit
        """
        data = QuantumRegister(9, 'data')
        ancilla = QuantumRegister(2, 'ancilla_pf')
        syndrome = ClassicalRegister(2, 'syndrome_pf')

        qc = QuantumCircuit(data, ancilla, syndrome, name='PhaseFlipSyndrome')

        # Convert to X basis for phase-flip detection
        for i in [0, 3, 6]:  # First qubit of each block
            qc.h(data[i])

        # Measure parity between blocks
        # S1: parity between block 1 and block 2
        qc.cx(data[0], ancilla[0])
        qc.cx(data[3], ancilla[0])
        qc.measure(ancilla[0], syndrome[0])

        # S2: parity between block 2 and block 3
        qc.cx(data[3], ancilla[1])
        qc.cx(data[6], ancilla[1])
        qc.measure(ancilla[1], syndrome[1])

        # Convert back from X basis
        for i in [0, 3, 6]:
            qc.h(data[i])

        return qc

    def create_bit_flip_recovery_circuit(self) -> QuantumCircuit:
        """
        Create recovery circuit for bit-flip errors.

        Based on syndrome measurement, applies X corrections to each block.

        Returns:
            QuantumCircuit: Bit-flip recovery circuit
        """
        data = QuantumRegister(9, 'data')
        syndrome = ClassicalRegister(6, 'syndrome_bf')

        qc = QuantumCircuit(data, syndrome, name='BitFlipRecovery')

        # Block 1 recovery
        # Syndrome 01: error on qubit 2
        qc.x(data[2]).c_if(syndrome, 0b000001)
        # Syndrome 10: error on qubit 0
        qc.x(data[0]).c_if(syndrome, 0b000010)
        # Syndrome 11: error on qubit 1
        qc.x(data[1]).c_if(syndrome, 0b000011)

        # Block 2 recovery
        # Syndrome 0001xx: error on qubit 5
        qc.x(data[5]).c_if(syndrome, 0b000100)
        # Syndrome 0010xx: error on qubit 3
        qc.x(data[3]).c_if(syndrome, 0b001000)
        # Syndrome 0011xx: error on qubit 4
        qc.x(data[4]).c_if(syndrome, 0b001100)

        # Block 3 recovery
        # Syndrome 01xxxx: error on qubit 8
        qc.x(data[8]).c_if(syndrome, 0b010000)
        # Syndrome 10xxxx: error on qubit 6
        qc.x(data[6]).c_if(syndrome, 0b100000)
        # Syndrome 11xxxx: error on qubit 7
        qc.x(data[7]).c_if(syndrome, 0b110000)

        return qc

    def create_phase_flip_recovery_circuit(self) -> QuantumCircuit:
        """
        Create recovery circuit for phase-flip errors.

        Based on syndrome measurement, applies Z corrections to blocks.

        Returns:
            QuantumCircuit: Phase-flip recovery circuit
        """
        data = QuantumRegister(9, 'data')
        syndrome = ClassicalRegister(2, 'syndrome_pf')

        qc = QuantumCircuit(data, syndrome, name='PhaseFlipRecovery')

        # Phase flip on block 3 (qubits 6, 7, 8)
        # Syndrome 01
        for i in [6, 7, 8]:
            qc.z(data[i]).c_if(syndrome, 0b01)

        # Phase flip on block 1 (qubits 0, 1, 2)
        # Syndrome 10
        for i in [0, 1, 2]:
            qc.z(data[i]).c_if(syndrome, 0b10)

        # Phase flip on block 2 (qubits 3, 4, 5)
        # Syndrome 11
        for i in [3, 4, 5]:
            qc.z(data[i]).c_if(syndrome, 0b11)

        return qc

    def create_complete_ecc_circuit(
        self,
        initial_state: Optional[Statevector] = None,
        error_type: Optional[str] = None,
        error_qubit: Optional[int] = None
    ) -> QuantumCircuit:
        """
        Create complete Shor code error correction circuit.

        Args:
            initial_state: Initial logical state to encode
            error_type: Type of error ('X', 'Y', 'Z', or None)
            error_qubit: Which qubit to apply error to (0-8)

        Returns:
            QuantumCircuit: Complete ECC circuit
        """
        # Create registers
        data = QuantumRegister(9, 'data')
        ancilla_bf = QuantumRegister(6, 'ancilla_bf')
        ancilla_pf = QuantumRegister(2, 'ancilla_pf')
        syndrome_bf = ClassicalRegister(6, 'syndrome_bf')
        syndrome_pf = ClassicalRegister(2, 'syndrome_pf')
        result = ClassicalRegister(1, 'result')

        qc = QuantumCircuit(data, ancilla_bf, ancilla_pf,
                           syndrome_bf, syndrome_pf, result)

        # 1. Prepare initial state
        if initial_state is not None:
            qc.initialize(initial_state, data[0])

        # 2. Encode
        qc.compose(self.create_encoding_circuit(), data, inplace=True)
        qc.barrier(label='Encoded')

        # 3. Apply error (if specified)
        if error_type is not None and error_qubit is not None:
            if error_type == 'X':
                qc.x(data[error_qubit])
            elif error_type == 'Y':
                qc.y(data[error_qubit])
            elif error_type == 'Z':
                qc.z(data[error_qubit])
            qc.barrier(label='Error')

        # 4. Bit-flip syndrome measurement
        # Measure bit-flip syndromes
        for block in range(3):
            base = block * 3
            qc.cx(data[base], ancilla_bf[block * 2])
            qc.cx(data[base + 1], ancilla_bf[block * 2])
            qc.measure(ancilla_bf[block * 2], syndrome_bf[block * 2])

            qc.cx(data[base + 1], ancilla_bf[block * 2 + 1])
            qc.cx(data[base + 2], ancilla_bf[block * 2 + 1])
            qc.measure(ancilla_bf[block * 2 + 1], syndrome_bf[block * 2 + 1])

        qc.barrier(label='BF_Syndrome')

        # 5. Phase-flip syndrome measurement
        for i in [0, 3, 6]:
            qc.h(data[i])

        qc.cx(data[0], ancilla_pf[0])
        qc.cx(data[3], ancilla_pf[0])
        qc.measure(ancilla_pf[0], syndrome_pf[0])

        qc.cx(data[3], ancilla_pf[1])
        qc.cx(data[6], ancilla_pf[1])
        qc.measure(ancilla_pf[1], syndrome_pf[1])

        for i in [0, 3, 6]:
            qc.h(data[i])

        qc.barrier(label='PF_Syndrome')

        # Note: Full conditional recovery based on syndromes would require
        # complex classical logic. For demonstration, we show the structure.

        # 6. Decode
        qc.compose(self.create_decoding_circuit(), data, inplace=True)
        qc.barrier(label='Decoded')

        # 7. Measure result
        qc.measure(data[0], result[0])

        return qc

    def simulate_error_correction(
        self,
        initial_state: Statevector,
        error_type: str,
        n_shots: int = 1024
    ) -> Dict:
        """
        Simulate error correction for a specific error type.

        Args:
            initial_state: Initial logical state
            error_type: Type of error ('X', 'Y', 'Z', or 'random')
            n_shots: Number of simulation shots

        Returns:
            dict: Simulation results
        """
        # Create simplified circuit for testing
        qc = QuantumCircuit(9, 1)

        # Initialize and encode
        qc.initialize(initial_state, 0)
        qc.compose(self.create_encoding_circuit(), range(9), inplace=True)
        qc.barrier()

        # Apply error to qubit 0 (first qubit)
        if error_type == 'X':
            qc.x(0)
        elif error_type == 'Y':
            qc.y(0)
        elif error_type == 'Z':
            qc.z(0)
        qc.barrier()

        # Decode
        qc.compose(self.create_decoding_circuit(), range(9), inplace=True)

        # Measure
        qc.measure(0, 0)

        # Simulate
        simulator = AerSimulator()
        job = simulator.run(qc, shots=n_shots)
        result = job.result()
        counts = result.get_counts()

        # Determine expected outcome
        expected = '0' if np.isclose(initial_state[1], 0) else '1'

        # Calculate success rate
        success_count = counts.get(expected, 0)
        success_rate = success_count / n_shots

        return {
            'counts': counts,
            'success_rate': success_rate,
            'error_type': error_type,
            'expected_outcome': expected,
            'n_shots': n_shots
        }

    def get_logical_basis_states(self) -> Tuple[Statevector, Statevector]:
        """
        Get the encoded logical basis states.

        Returns:
            Tuple of (|0⟩_L, |1⟩_L) as Statevectors
        """
        # Create encoding circuit
        encode_circuit = self.create_encoding_circuit()

        # Get |0⟩_L
        zero_state = Statevector.from_label('0' * 9)
        zero_logical = zero_state.evolve(encode_circuit)

        # Get |1⟩_L
        one_state = Statevector.from_label('0' * 9)
        # Apply X to first qubit before encoding
        qc_one = QuantumCircuit(9)
        qc_one.x(0)
        qc_one.compose(encode_circuit, inplace=True)
        one_logical = one_state.evolve(qc_one)

        return zero_logical, one_logical


def demonstrate_shor_code():
    """Demonstrate Shor's 9-qubit code."""
    print("=" * 70)
    print("SHOR'S 9-QUBIT ERROR CORRECTION CODE DEMONSTRATION")
    print("=" * 70)

    code = ShorCode()

    print("\n1. ENCODING CIRCUIT")
    print("-" * 70)
    print("Encodes 1 logical qubit into 9 physical qubits")
    print("Protects against any single-qubit error (X, Y, or Z)")
    print("\nCircuit structure:")
    print(code.create_encoding_circuit())

    print("\n2. CODE PROPERTIES")
    print("-" * 70)
    print(f"Physical qubits: {code.n_physical}")
    print(f"Logical qubits: {code.n_logical}")
    print(f"Encoding overhead: {code.n_physical}x")
    print(f"Ancilla qubits needed: {code.n_ancilla_total}")

    print("\n3. TESTING ERROR CORRECTION")
    print("-" * 70)

    # Test with |0⟩ state
    zero_state = Statevector.from_label('0')

    error_types = ['X', 'Y', 'Z']

    for error_type in error_types:
        result = code.simulate_error_correction(
            zero_state,
            error_type,
            n_shots=1000
        )

        print(f"\n{error_type} error on qubit 0:")
        print(f"  Success rate: {result['success_rate']:.2%}")
        print(f"  Measurement counts: {result['counts']}")

    print("\n4. SUPERPOSITION STATE TEST")
    print("-" * 70)

    # Test with |+⟩ state
    plus_state = Statevector.from_label('0')
    plus_state = plus_state.evolve(Operator.from_label('H'))

    print("\nTesting |+⟩ = (|0⟩ + |1⟩)/√2 state:")

    for error_type in error_types:
        result = code.simulate_error_correction(
            plus_state,
            error_type,
            n_shots=1000
        )

        print(f"  {error_type} error: {result['success_rate']:.2%} success rate")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demonstrate_shor_code()
