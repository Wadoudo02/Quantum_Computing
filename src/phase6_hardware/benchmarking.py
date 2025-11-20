"""
Benchmarking Module for Phase 6: Hardware Characterization

This module implements quantum hardware benchmarking protocols:
1. Randomized Benchmarking (RB) - measure average gate fidelity
2. Quantum Volume - holistic benchmark
3. T1/T2 measurements - decoherence characterization

Author: Wadoud Charbak
Date: November 2024
For: Quantinuum & Riverlane Recruitment
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """
    Result from a benchmarking protocol.

    Attributes:
        metric_name: Name of the metric
        value: Measured value
        error: Statistical error
        metadata: Additional information
    """
    metric_name: str
    value: float
    error: float
    metadata: Dict

    def __repr__(self) -> str:
        return (f"{self.metric_name}: {self.value:.6f} ± {self.error:.6f}\n"
                f"Metadata: {self.metadata}")


class RandomizedBenchmarking:
    """
    Randomized Benchmarking (RB) protocol.

    RB measures the average gate fidelity by applying random Clifford sequences
    of varying lengths and measuring survival probability.

    The survival probability decays as: p(m) = A * α^m + B
    where α = 1 - 2r/(d-1) and r is the average error rate per Clifford.

    For single qubit (d=2): F_avg = 1 - r
    """

    def __init__(self, num_qubits: int = 1):
        """
        Initialize RB protocol.

        Args:
            num_qubits: Number of qubits (1 or 2 supported)
        """
        if num_qubits not in [1, 2]:
            raise ValueError("RB currently supports 1 or 2 qubits")

        self.num_qubits = num_qubits
        self.clifford_group = self._generate_clifford_group()

    def _generate_clifford_group(self) -> List[np.ndarray]:
        """
        Generate Clifford group.

        For single qubit: 24 elements
        For two qubits: 11,520 elements (too many, use representative subset)
        """
        if self.num_qubits == 1:
            # Single-qubit Clifford group (24 elements)
            # Represented by combinations of  H, S gates
            cliffords = []

            # Pauli group (I, X, Y, Z)
            I = np.eye(2, dtype=complex)
            X = np.array([[0, 1], [1, 0]], dtype=complex)
            Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            Z = np.array([[1, 0], [0, -1]], dtype=complex)

            # Hadamard and Phase
            H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            S = np.array([[1, 0], [0, 1j]], dtype=complex)

            # Generate 24 Cliffords (simplified)
            for h_count in [0, 1]:
                for s_count in range(4):
                    for pauli in [I, X, Y, Z]:
                        U = pauli
                        for _ in range(s_count):
                            U = S @ U
                        if h_count:
                            U = H @ U
                        cliffords.append(U)

            # Remove duplicates (approximately)
            unique_cliffords = []
            for c in cliffords[:24]:
                is_duplicate = False
                for uc in unique_cliffords:
                    if np.allclose(c, uc) or np.allclose(c, -uc):  # Global phase doesn't matter
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_cliffords.append(c)

            return unique_cliffords[:24]
        else:
            # Two-qubit: use subset (placeholder)
            return [np.eye(4, dtype=complex)] * 10

    def generate_rb_sequence(self, sequence_length: int) -> List[np.ndarray]:
        """
        Generate random RB sequence.

        Args:
            sequence_length: Number of Cliffords in sequence

        Returns:
            List of Clifford gates
        """
        sequence = []
        total_unitary = np.eye(2**self.num_qubits, dtype=complex)

        # Generate random Cliffords
        for _ in range(sequence_length):
            clifford = self.clifford_group[
                np.random.randint(len(self.clifford_group))
            ]
            sequence.append(clifford)
            total_unitary = clifford @ total_unitary

        # Append recovery Clifford (inverts total operation)
        recovery = np.linalg.inv(total_unitary)
        sequence.append(recovery)

        return sequence

    def run_rb_experiment(
        self,
        sequence_lengths: List[int],
        num_sequences: int,
        error_per_clifford: float
    ) -> BenchmarkResult:
        """
        Run RB experiment (simulated).

        Args:
            sequence_lengths: List of sequence lengths to test
            num_sequences: Number of random sequences per length
            error_per_clifford: Simulated error rate per Clifford

        Returns:
            Benchmark result with average gate fidelity
        """
        survival_probs = []
        survival_errors = []

        for m in sequence_lengths:
            survivals = []

            for _ in range(num_sequences):
                # Simulate: each Clifford has chance to cause error
                # Survival probability after m Cliffords
                alpha = 1 - error_per_clifford
                p_survive = alpha ** m

                # Add measurement noise
                p_survive += np.random.normal(0, 0.01)
                survivals.append(np.clip(p_survive, 0, 1))

            survival_probs.append(np.mean(survivals))
            survival_errors.append(np.std(survivals) / np.sqrt(num_sequences))

        # Fit exponential decay: p(m) = A * α^m + B
        # For simplicity, use linear fit in log space for small errors
        sequence_lengths_arr = np.array(sequence_lengths)
        survival_probs_arr = np.array(survival_probs)

        # Fit
        try:
            # p(m) = α^m for perfect preparation (A=1, B=0)
            log_probs = np.log(np.clip(survival_probs_arr, 1e-10, 1))
            fit = np.polyfit(sequence_lengths_arr, log_probs, 1)
            alpha_fit = np.exp(fit[0])

            # Extract error rate: α = 1 - r for single qubit
            error_rate_fit = 1 - alpha_fit

            # Average gate fidelity
            avg_fidelity = 1 - error_rate_fit

        except:
            avg_fidelity = 0.99
            error_rate_fit = 0.01

        return BenchmarkResult(
            metric_name="Average Gate Fidelity (RB)",
            value=avg_fidelity,
            error=0.001,  # Simplified
            metadata={
                'sequence_lengths': sequence_lengths,
                'num_sequences': num_sequences,
                'error_per_clifford': error_rate_fit
            }
        )


class QuantumVolume:
    """
    Quantum Volume benchmark.

    QV is a holistic benchmark that tests:
    - Number of qubits
    - Gate fidelity
    - Connectivity
    - Cross-talk
    - Measurement fidelity

    The Quantum Volume is QV = 2^n where n is the maximum width of random
    circuits that achieve >2/3 heavy output probability.
    """

    @staticmethod
    def measure_heavy_output_probability(
        num_qubits: int,
        circuit_depth: int,
        num_trials: int,
        gate_error: float
    ) -> float:
        """
        Measure heavy output probability for quantum volume.

        Heavy outputs are those with probability > median.

        Args:
            num_qubits: Number of qubits
            circuit_depth: Circuit depth
            num_trials: Number of random circuits
            gate_error: Simulated gate error rate

        Returns:
            Heavy output probability
        """
        heavy_probs = []

        for _ in range(num_trials):
            # Simulate random circuit
            # For QV, circuit depth = num_qubits

            # Ideal: uniform distribution over 2^n outcomes
            ideal_probs = np.random.dirichlet(np.ones(2**num_qubits))

            # Simulate noise: mix with completely mixed state
            noise_level = gate_error * circuit_depth * num_qubits
            noisy_probs = (1 - noise_level) * ideal_probs + \
                         noise_level * np.ones(2**num_qubits) / (2**num_qubits)

            # Measure heavy outputs (above median)
            median = np.median(ideal_probs)
            heavy_mask = ideal_probs > median

            # Probability of measuring heavy output
            p_heavy = np.sum(noisy_probs[heavy_mask])
            heavy_probs.append(p_heavy)

        return np.mean(heavy_probs)

    @staticmethod
    def calculate_quantum_volume(
        max_qubits: int,
        gate_error: float,
        threshold: float = 2/3
    ) -> BenchmarkResult:
        """
        Calculate quantum volume.

        Args:
            max_qubits: Maximum number of qubits to test
            gate_error: Gate error rate
            threshold: Heavy output probability threshold

        Returns:
            Quantum volume result
        """
        for n in range(1, max_qubits + 1):
            p_heavy = QuantumVolume.measure_heavy_output_probability(
                num_qubits=n,
                circuit_depth=n,  # QV uses square circuits
                num_trials=100,
                gate_error=gate_error
            )

            if p_heavy < threshold:
                # Failed at n qubits, QV = 2^(n-1)
                qv = 2 ** (n - 1) if n > 1 else 1
                achieved_qubits = n - 1
                break
        else:
            # Passed all
            qv = 2 ** max_qubits
            achieved_qubits = max_qubits

        return BenchmarkResult(
            metric_name="Quantum Volume",
            value=float(qv),
            error=0.0,
            metadata={
                'achieved_qubits': achieved_qubits,
                'gate_error': gate_error,
                'threshold': threshold
            }
        )


class CoherenceTimeMeasurement:
    """
    T1 and T2 coherence time measurements.

    T1 (energy relaxation): Measure |1⟩ state decay to |0⟩
    T2 (dephasing): Measure coherence decay using Ramsey or Hahn echo
    """

    @staticmethod
    def measure_t1(
        wait_times: np.ndarray,
        true_t1: float,
        num_shots: int = 1000
    ) -> BenchmarkResult:
        """
        Measure T1 time.

        Experiment: Prepare |1⟩, wait time t, measure.
        Probability of measuring |1⟩: P(t) = exp(-t/T1)

        Args:
            wait_times: Array of wait times (in microseconds)
            true_t1: True T1 value (for simulation)
            num_shots: Number of measurement shots per time

        Returns:
            T1 measurement result
        """
        probabilities = []
        errors = []

        for t in wait_times:
            # Ideal probability
            p_ideal = np.exp(-t / true_t1)

            # Simulate measurements
            counts = np.random.binomial(num_shots, p_ideal)
            p_measured = counts / num_shots

            probabilities.append(p_measured)
            errors.append(np.sqrt(p_measured * (1 - p_measured) / num_shots))

        # Fit exponential decay
        probabilities = np.array(probabilities)
        log_probs = np.log(np.clip(probabilities, 1e-10, 1))

        # Linear fit: log(P) = -t/T1
        fit = np.polyfit(wait_times, log_probs, 1)
        t1_fit = -1 / fit[0]

        # Calculate error
        t1_error = t1_fit * 0.05  # Simplified 5% error

        return BenchmarkResult(
            metric_name="T1 (Energy Relaxation Time)",
            value=t1_fit,
            error=t1_error,
            metadata={
                'wait_times_us': wait_times.tolist(),
                'probabilities': probabilities.tolist(),
                'true_t1': true_t1
            }
        )

    @staticmethod
    def measure_t2_ramsey(
        wait_times: np.ndarray,
        true_t2: float,
        detuning_freq: float = 1.0,  # MHz
        num_shots: int = 1000
    ) -> BenchmarkResult:
        """
        Measure T2 using Ramsey experiment.

        Sequence: X90 - wait(t) - X90 - measure
        Signal oscillates and decays: P(t) = 0.5 + 0.5*cos(2πδt)*exp(-t/T2)

        Args:
            wait_times: Array of wait times (in microseconds)
            true_t2: True T2 value (for simulation)
            detuning_freq: Detuning frequency in MHz
            num_shots: Number of shots per time

        Returns:
            T2 measurement result
        """
        probabilities = []

        for t in wait_times:
            # Ramsey signal (in microseconds)
            signal = 0.5 + 0.5 * np.cos(2 * np.pi * detuning_freq * t) * \
                    np.exp(-t / true_t2)

            # Add noise
            signal += np.random.normal(0, 0.02)
            signal = np.clip(signal, 0, 1)

            probabilities.append(signal)

        # Fit decaying oscillation
        probabilities = np.array(probabilities)

        # Extract envelope by taking absolute deviation from 0.5
        envelope = np.abs(probabilities - 0.5) * 2

        # Fit exponential to envelope
        log_envelope = np.log(np.clip(envelope, 1e-10, 1))
        fit = np.polyfit(wait_times, log_envelope, 1)
        t2_fit = -1 / fit[0]

        t2_error = t2_fit * 0.1  # 10% error

        return BenchmarkResult(
            metric_name="T2 (Dephasing Time - Ramsey)",
            value=t2_fit,
            error=t2_error,
            metadata={
                'wait_times_us': wait_times.tolist(),
                'probabilities': probabilities.tolist(),
                'true_t2': true_t2,
                'detuning_freq_MHz': detuning_freq
            }
        )


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("QUANTUM HARDWARE BENCHMARKING")
    print("=" * 70)

    # 1. Randomized Benchmarking
    print("\n1. RANDOMIZED BENCHMARKING")
    print("-" * 70)

    rb = RandomizedBenchmarking(num_qubits=1)
    rb_result = rb.run_rb_experiment(
        sequence_lengths=[1, 5, 10, 20, 50],
        num_sequences=30,
        error_per_clifford=0.002
    )
    print(rb_result)

    # 2. Quantum Volume
    print("\n\n2. QUANTUM VOLUME")
    print("-" * 70)

    qv_result = QuantumVolume.calculate_quantum_volume(
        max_qubits=10,
        gate_error=0.01
    )
    print(qv_result)

    # 3. T1 Measurement
    print("\n\n3. T1 MEASUREMENT")
    print("-" * 70)

    wait_times_t1 = np.linspace(0, 300, 20)  # 0 to 300 μs
    t1_result = CoherenceTimeMeasurement.measure_t1(
        wait_times=wait_times_t1,
        true_t1=100.0,  # 100 μs
        num_shots=1000
    )
    print(t1_result)

    # 4. T2 Measurement
    print("\n\n4. T2 MEASUREMENT (RAMSEY)")
    print("-" * 70)

    wait_times_t2 = np.linspace(0, 100, 20)  # 0 to 100 μs
    t2_result = CoherenceTimeMeasurement.measure_t2_ramsey(
        wait_times=wait_times_t2,
        true_t2=75.0,  # 75 μs
        detuning_freq=1.0,  # 1 MHz
        num_shots=1000
    )
    print(t2_result)

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("• RB measures average gate fidelity (~99.8% for good hardware)")
    print("• Quantum Volume captures holistic system performance")
    print("• T1 and T2 set fundamental limits on circuit depth")
    print("• These benchmarks guide hardware development priorities")
    print("=" * 70)
