"""
3-Qubit Bit-Flip Error Correction Code

This module implements the simplest quantum error correction code: the 3-qubit
bit-flip code. It protects against single bit-flip (X) errors by encoding one
logical qubit into three physical qubits.

Encoding:
    |0⟩_L = |000⟩
    |1⟩_L = |111⟩

The code can detect and correct a single bit-flip error on any of the three qubits
using syndrome measurement without collapsing the logical state.

Author: Quantum Computing Learning Project
Phase: 5 - Error Correction
"""

import numpy as np
from typing import Tuple, List, Optional
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, pauli_error, depolarizing_error


class BitFlipCode:
    """
    Implementation of the 3-qubit bit-flip error correction code.

    This code encodes 1 logical qubit into 3 physical qubits using redundancy.
    It can correct any single bit-flip (X) error using syndrome measurement.

    Attributes:
        n_physical (int): Number of physical qubits (3)
        n_logical (int): Number of logical qubits (1)
        encoding_circuit (QuantumCircuit): Circuit to encode logical state
        syndrome_circuit (QuantumCircuit): Circuit to measure error syndrome
        recovery_circuit (QuantumCircuit): Circuit to apply recovery operations
    """

    def __init__(self):
        """Initialize the 3-qubit bit-flip code."""
        self.n_physical = 3
        self.n_logical = 1
        self.n_ancilla = 2  # Two ancilla qubits for syndrome measurement

        # Create the encoding circuit
        self.encoding_circuit = self._create_encoding_circuit()

    def _create_encoding_circuit(self) -> QuantumCircuit:
        """
        Create the encoding circuit for the 3-qubit bit-flip code.

        Encodes |ψ⟩ = α|0⟩ + β|1⟩ into |ψ⟩_L = α|000⟩ + β|111⟩

        Returns:
            QuantumCircuit: The encoding circuit
        """
        qc = QuantumCircuit(3, name='Encode')

        # Start with logical qubit in qubit 0
        # Encode by copying the state to qubits 1 and 2
        qc.cx(0, 1)  # Copy to qubit 1
        qc.cx(0, 2)  # Copy to qubit 2

        # Now: |0⟩ → |000⟩ and |1⟩ → |111⟩
        # Superposition preserved: α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩

        return qc

    def encode(self, initial_state: Optional[Statevector] = None) -> QuantumCircuit:
        """
        Create a circuit that encodes a logical qubit.

        Args:
            initial_state: Optional initial state to prepare. If None, starts in |0⟩

        Returns:
            QuantumCircuit: Complete circuit with state preparation and encoding
        """
        qc = QuantumCircuit(3, name='BitFlipEncode')

        # Prepare initial state if provided
        if initial_state is not None:
            qc.initialize(initial_state, 0)

        # Apply encoding
        qc.compose(self.encoding_circuit, inplace=True)

        return qc

    def create_syndrome_circuit(self) -> QuantumCircuit:
        """
        Create syndrome measurement circuit.

        Measures two syndromes:
        - S1 = Z0 ⊗ Z1 (parity of qubits 0 and 1)
        - S2 = Z1 ⊗ Z2 (parity of qubits 1 and 2)

        Syndrome outcomes:
        - 00: No error
        - 01: Error on qubit 2
        - 10: Error on qubit 0
        - 11: Error on qubit 1

        Returns:
            QuantumCircuit: Syndrome measurement circuit
        """
        # 3 data qubits + 2 ancilla qubits for syndrome measurement
        data = QuantumRegister(3, 'data')
        ancilla = QuantumRegister(2, 'ancilla')
        syndrome = ClassicalRegister(2, 'syndrome')

        qc = QuantumCircuit(data, ancilla, syndrome, name='SyndromeMeasure')

        # Measure S1 = parity of qubits 0 and 1
        qc.cx(data[0], ancilla[0])
        qc.cx(data[1], ancilla[0])
        qc.measure(ancilla[0], syndrome[0])

        # Measure S2 = parity of qubits 1 and 2
        qc.cx(data[1], ancilla[1])
        qc.cx(data[2], ancilla[1])
        qc.measure(ancilla[1], syndrome[1])

        return qc

    def create_recovery_circuit(self) -> QuantumCircuit:
        """
        Create recovery circuit with conditional corrections.

        Based on syndrome measurement, applies correction:
        - syndrome = 00 (0): No correction
        - syndrome = 01 (1): X on qubit 2
        - syndrome = 10 (2): X on qubit 0
        - syndrome = 11 (3): X on qubit 1

        Returns:
            QuantumCircuit: Recovery circuit
        """
        data = QuantumRegister(3, 'data')
        syndrome = ClassicalRegister(2, 'syndrome')

        qc = QuantumCircuit(data, syndrome, name='Recovery')

        # Syndrome = 01: Error on qubit 2
        qc.x(data[2]).c_if(syndrome, 1)

        # Syndrome = 10: Error on qubit 0
        qc.x(data[0]).c_if(syndrome, 2)

        # Syndrome = 11: Error on qubit 1
        qc.x(data[1]).c_if(syndrome, 3)

        return qc

    def decode(self) -> QuantumCircuit:
        """
        Create decoding circuit (inverse of encoding).

        Returns:
            QuantumCircuit: Decoding circuit
        """
        qc = QuantumCircuit(3, name='Decode')

        # Inverse of encoding: reverse the CNOTs
        qc.cx(0, 2)
        qc.cx(0, 1)

        return qc

    def create_complete_ecc_circuit(
        self,
        initial_state: Optional[Statevector] = None,
        error_qubit: Optional[int] = None,
        include_recovery: bool = True
    ) -> QuantumCircuit:
        """
        Create complete error correction circuit.

        Args:
            initial_state: Initial logical state to encode
            error_qubit: Which qubit to flip (0, 1, or 2). None for no error.
            include_recovery: Whether to include syndrome measurement and recovery

        Returns:
            QuantumCircuit: Complete ECC circuit
        """
        # Create registers
        data = QuantumRegister(3, 'data')
        ancilla = QuantumRegister(2, 'ancilla')
        syndrome = ClassicalRegister(2, 'syndrome')
        result = ClassicalRegister(1, 'result')

        qc = QuantumCircuit(data, ancilla, syndrome, result)

        # 1. Prepare initial state
        if initial_state is not None:
            qc.initialize(initial_state, data[0])

        # 2. Encode
        qc.compose(self.encoding_circuit, data, inplace=True)
        qc.barrier()

        # 3. Introduce error (if specified)
        if error_qubit is not None:
            qc.x(data[error_qubit])
            qc.barrier()

        # 4. Syndrome measurement and recovery
        if include_recovery:
            # Syndrome measurement
            qc.cx(data[0], ancilla[0])
            qc.cx(data[1], ancilla[0])
            qc.measure(ancilla[0], syndrome[0])

            qc.cx(data[1], ancilla[1])
            qc.cx(data[2], ancilla[1])
            qc.measure(ancilla[1], syndrome[1])

            qc.barrier()

            # Recovery
            qc.x(data[2]).c_if(syndrome, 1)
            qc.x(data[0]).c_if(syndrome, 2)
            qc.x(data[1]).c_if(syndrome, 3)

            qc.barrier()

        # 5. Decode
        qc.compose(self.decode(), data, inplace=True)

        # 6. Measure result
        qc.measure(data[0], result[0])

        return qc

    def simulate_error_correction(
        self,
        initial_state: Statevector,
        error_probability: float = 0.1,
        n_shots: int = 1024
    ) -> dict:
        """
        Simulate error correction with random bit-flip errors.

        Args:
            initial_state: Initial logical state
            error_probability: Probability of bit-flip on each qubit
            n_shots: Number of simulation shots

        Returns:
            dict: Simulation results including success rates
        """
        # Create circuit
        data = QuantumRegister(3, 'data')
        ancilla = QuantumRegister(2, 'ancilla')
        syndrome = ClassicalRegister(2, 'syndrome')
        result = ClassicalRegister(1, 'result')

        qc = QuantumCircuit(data, ancilla, syndrome, result)

        # Initialize and encode
        qc.initialize(initial_state, data[0])
        qc.compose(self.encoding_circuit, data, inplace=True)
        qc.barrier()

        # Add noise channel (bit-flip errors will be added via noise model)
        qc.id(data[0])
        qc.id(data[1])
        qc.id(data[2])
        qc.barrier()

        # Syndrome measurement
        qc.cx(data[0], ancilla[0])
        qc.cx(data[1], ancilla[0])
        qc.measure(ancilla[0], syndrome[0])

        qc.cx(data[1], ancilla[1])
        qc.cx(data[2], ancilla[1])
        qc.measure(ancilla[1], syndrome[1])
        qc.barrier()

        # Recovery
        qc.x(data[2]).c_if(syndrome, 1)
        qc.x(data[0]).c_if(syndrome, 2)
        qc.x(data[1]).c_if(syndrome, 3)
        qc.barrier()

        # Decode and measure
        qc.compose(self.decode(), data, inplace=True)
        qc.measure(data[0], result[0])

        # Create noise model
        noise_model = NoiseModel()
        error = pauli_error([('X', error_probability), ('I', 1 - error_probability)])
        noise_model.add_all_qubit_quantum_error(error, ['id'])

        # Simulate
        simulator = AerSimulator(noise_model=noise_model)
        job = simulator.run(qc, shots=n_shots)
        result = job.result()
        counts = result.get_counts()

        # Analyze results
        expected_outcome = '0' if np.isclose(initial_state[1], 0) else '1'

        # Extract syndromes and results
        syndrome_counts = {}
        success_count = 0

        for outcome, count in counts.items():
            # Format: 'result syndrome[1] syndrome[0]'
            parts = outcome.split()
            if len(parts) >= 2:
                measured_result = parts[0]
                syndrome_bits = parts[1] if len(parts) > 1 else '00'
            else:
                measured_result = outcome[0]
                syndrome_bits = outcome[1:] if len(outcome) > 1 else '00'

            syndrome_counts[syndrome_bits] = syndrome_counts.get(syndrome_bits, 0) + count

            if measured_result == expected_outcome:
                success_count += count

        success_rate = success_count / n_shots

        return {
            'counts': counts,
            'syndrome_counts': syndrome_counts,
            'success_rate': success_rate,
            'error_probability': error_probability,
            'expected_outcome': expected_outcome,
            'n_shots': n_shots
        }

    def analyze_threshold(
        self,
        initial_state: Statevector,
        error_rates: List[float],
        n_shots: int = 1024
    ) -> Tuple[List[float], List[float]]:
        """
        Analyze error correction performance across different error rates.

        Args:
            initial_state: Initial logical state
            error_rates: List of error probabilities to test
            n_shots: Number of shots per error rate

        Returns:
            Tuple of (error_rates, success_rates)
        """
        success_rates = []

        for p_error in error_rates:
            result = self.simulate_error_correction(
                initial_state, p_error, n_shots
            )
            success_rates.append(result['success_rate'])

        return error_rates, success_rates

    def get_syndrome_lookup(self) -> dict:
        """
        Get syndrome to error location mapping.

        Returns:
            dict: Syndrome -> error location mapping
        """
        return {
            '00': 'No error',
            '01': 'Error on qubit 2',
            '10': 'Error on qubit 0',
            '11': 'Error on qubit 1'
        }


def demonstrate_bit_flip_code():
    """Demonstrate the 3-qubit bit-flip code."""
    print("=" * 70)
    print("3-QUBIT BIT-FLIP ERROR CORRECTION CODE DEMONSTRATION")
    print("=" * 70)

    code = BitFlipCode()

    print("\n1. ENCODING CIRCUIT")
    print("-" * 70)
    print("Encodes: |ψ⟩ = α|0⟩ + β|1⟩ → |ψ⟩_L = α|000⟩ + β|111⟩")
    print(code.encoding_circuit)

    print("\n2. SYNDROME MEASUREMENT")
    print("-" * 70)
    print("Syndrome lookup table:")
    for syndrome, location in code.get_syndrome_lookup().items():
        print(f"  {syndrome}: {location}")

    print("\n3. TESTING ERROR CORRECTION")
    print("-" * 70)

    # Test state |+⟩ = (|0⟩ + |1⟩)/√2
    plus_state = Statevector.from_label('0')
    plus_state = plus_state.evolve(Operator.from_label('H'))

    # Test with error on each qubit
    for error_qubit in range(3):
        qc = code.create_complete_ecc_circuit(
            initial_state=plus_state,
            error_qubit=error_qubit
        )

        # Simulate
        simulator = AerSimulator()
        job = simulator.run(qc, shots=1000)
        result = job.result()
        counts = result.get_counts()

        print(f"\nError on qubit {error_qubit}:")
        print(f"  Measurement results: {counts}")

    print("\n4. ERROR RATE ANALYSIS")
    print("-" * 70)

    error_rates = [0.01, 0.05, 0.1, 0.15, 0.2]
    rates, success = code.analyze_threshold(plus_state, error_rates, n_shots=1000)

    print("\nError Rate | Success Rate")
    print("-" * 30)
    for p, s in zip(rates, success):
        print(f"  {p:6.2%}   | {s:6.2%}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demonstrate_bit_flip_code()
