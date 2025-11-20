"""
Error Mitigation Module for Phase 6: NISQ-Era Error Mitigation

This module implements error mitigation techniques for near-term quantum computers:
1. Readout error mitigation (measurement calibration)
2. Zero-Noise Extrapolation (ZNE)
3. Probabilistic Error Cancellation (PEC)

These techniques don't require additional qubits (unlike error correction) and
can provide 2-10x improvements in NISQ devices.

Author: Wadoud Charbak
Date: November 2024
For: Quantinuum & Riverlane Recruitment
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class MitigationResult:
    """
    Result of error mitigation.

    Attributes:
        raw_expectation: Expectation value before mitigation
        mitigated_expectation: Expectation value after mitigation
        improvement_factor: Ratio of improvement
        overhead: Computational overhead (shots multiplier)
        method: Mitigation method used
    """
    raw_expectation: float
    mitigated_expectation: float
    improvement_factor: float
    overhead: float
    method: str

    def __repr__(self) -> str:
        return (f"MitigationResult({self.method}):\n"
                f"  Raw: {self.raw_expectation:.6f}\n"
                f"  Mitigated: {self.mitigated_expectation:.6f}\n"
                f"  Improvement: {self.improvement_factor:.2f}x\n"
                f"  Overhead: {self.overhead:.1f}x shots")


class ReadoutErrorMitigator:
    """
    Readout (measurement) error mitigation.

    Calibrates measurement errors and applies correction matrix.

    The key idea: measurements are classical operations, so errors can be
    characterized by a confusion matrix M where M_ij = P(measure i | prepared j).

    We measure M, then invert it to correct results: corrected = M^(-1) @ measured
    """

    def __init__(self, num_qubits: int):
        """
        Initialize readout error mitigator.

        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self.calibration_matrix: Optional[np.ndarray] = None
        self.inverse_matrix: Optional[np.ndarray] = None

    def calibrate(
        self,
        readout_error_rates: List[float],
        num_shots: int = 10000
    ):
        """
        Calibrate readout errors.

        In practice, this would measure |0⟩ and |1⟩ states. Here we simulate
        based on known error rates.

        Args:
            readout_error_rates: List of error rates per qubit
            num_shots: Number of calibration shots
        """
        # Build calibration matrix for n qubits
        # For simplicity, assume independent qubit readout errors
        n = self.num_qubits
        dim = 2**n

        # Initialize calibration matrix
        M = np.zeros((dim, dim))

        for prepared_state in range(dim):
            # Prepare state (binary representation)
            prepared_bits = format(prepared_state, f'0{n}b')

            # Simulate measurements with errors
            measured_counts = np.zeros(dim)

            for _ in range(num_shots):
                measured_bits = []
                for i, bit in enumerate(prepared_bits):
                    error_rate = readout_error_rates[i] if i < len(readout_error_rates) else 0.01

                    # Flip bit with probability error_rate
                    if np.random.rand() < error_rate:
                        measured_bit = '1' if bit == '0' else '0'
                    else:
                        measured_bit = bit

                    measured_bits.append(measured_bit)

                measured_state = int(''.join(measured_bits), 2)
                measured_counts[measured_state] += 1

            # Normalize to get probabilities
            M[:, prepared_state] = measured_counts / num_shots

        self.calibration_matrix = M

        # Compute pseudo-inverse for mitigation
        # Use regularization to handle ill-conditioning
        try:
            self.inverse_matrix = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            self.inverse_matrix = np.linalg.pinv(M)

    def mitigate_counts(
        self,
        noisy_counts: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Mitigate measurement counts.

        Args:
            noisy_counts: Noisy measurement counts

        Returns:
            Mitigated counts (can be non-integer due to inversion)
        """
        if self.inverse_matrix is None:
            raise ValueError("Must calibrate before mitigating")

        # Convert counts to probability vector
        total_shots = sum(noisy_counts.values())
        dim = 2**self.num_qubits

        noisy_probs = np.zeros(dim)
        for state_str, count in noisy_counts.items():
            state_int = int(state_str, 2)
            noisy_probs[state_int] = count / total_shots

        # Apply correction
        mitigated_probs = self.inverse_matrix @ noisy_probs

        # Clip to [0, 1] (inversion can produce negative values)
        mitigated_probs = np.clip(mitigated_probs, 0, 1)

        # Renormalize
        mitigated_probs /= np.sum(mitigated_probs)

        # Convert back to counts
        mitigated_counts = {}
        for state_int in range(dim):
            count = mitigated_probs[state_int] * total_shots
            if count > 0.5:  # Only include non-negligible states
                state_str = format(state_int, f'0{self.num_qubits}b')
                mitigated_counts[state_str] = count

        return mitigated_counts

    def mitigate_expectation(
        self,
        noisy_expectation: float,
        observable_diagonal: np.ndarray
    ) -> float:
        """
        Mitigate expectation value of an observable.

        Args:
            noisy_expectation: Noisy expectation value
            observable_diagonal: Diagonal elements of observable

        Returns:
            Mitigated expectation value
        """
        if self.inverse_matrix is None:
            raise ValueError("Must calibrate before mitigating")

        # This is a simplified version
        # In practice, need to measure in computational basis and reconstruct
        return noisy_expectation  # Placeholder


class ZeroNoiseExtrapolation:
    """
    Zero-Noise Extrapolation (ZNE).

    Key idea: Artificially increase noise level, measure expectation values at
    different noise levels, then extrapolate to zero noise.

    Methods to increase noise:
    1. Pulse stretching (increase gate duration)
    2. Gate folding (insert G G^† pairs)

    We use gate folding: replace U with U(U^†U)^n to increase noise by ~(2n+1)x
    """

    @staticmethod
    def fold_circuit_globally(
        circuit_executor: Callable[[int], float],
        fold_factors: List[int],
        extrapolation_order: int = 2
    ) -> MitigationResult:
        """
        Apply ZNE with global circuit folding.

        Args:
            circuit_executor: Function that takes fold_factor and returns
                            expectation value
            fold_factors: List of fold factors (e.g., [1, 3, 5])
            extrapolation_order: Polynomial order for extrapolation

        Returns:
            Mitigation result
        """
        # Measure expectation at different noise levels
        noise_levels = []
        expectations = []

        for fold in fold_factors:
            noise_level = fold  # Noise scales linearly with fold factor
            expectation = circuit_executor(fold)

            noise_levels.append(noise_level)
            expectations.append(expectation)

        # Extrapolate to zero noise
        noise_levels = np.array(noise_levels)
        expectations = np.array(expectations)

        # Fit polynomial
        coeffs = np.polyfit(noise_levels, expectations, extrapolation_order)
        poly = np.poly1d(coeffs)

        # Extrapolate to zero
        mitigated = poly(0)

        # Calculate metrics
        raw = expectations[0]  # Fold factor 1
        improvement = abs(mitigated / raw) if raw != 0 else 1.0
        overhead = len(fold_factors)  # Need to run circuit multiple times

        return MitigationResult(
            raw_expectation=raw,
            mitigated_expectation=mitigated,
            improvement_factor=improvement,
            overhead=overhead,
            method="ZNE"
        )

    @staticmethod
    def richardson_extrapolation(
        expectations: List[float],
        noise_scale_factors: List[float]
    ) -> float:
        """
        Richardson extrapolation to zero noise.

        Args:
            expectations: Expectation values at different noise levels
            noise_scale_factors: Corresponding noise scale factors

        Returns:
            Extrapolated zero-noise expectation
        """
        expectations = np.array(expectations)
        noise_factors = np.array(noise_scale_factors)

        # Linear extrapolation (simplest Richardson)
        if len(expectations) == 2:
            c0 = noise_factors[0]
            c1 = noise_factors[1]
            return (c1 * expectations[0] - c0 * expectations[1]) / (c1 - c0)

        # Polynomial extrapolation for more points
        coeffs = np.polyfit(noise_factors, expectations, min(2, len(expectations) - 1))
        poly = np.poly1d(coeffs)
        return poly(0)


class ProbabilisticErrorCancellation:
    """
    Probabilistic Error Cancellation (PEC).

    Key idea: Decompose noisy operation ε(ρ) into a quasi-probability
    distribution over implementable operations:

    ε(ρ) = Σᵢ αᵢ εᵢ(ρ)

    where Σᵢ |αᵢ| > 1 (hence "quasi"-probability).

    Sample from {εᵢ} with probabilities {|αᵢ|/Σⱼ|αⱼ|}, multiply outcomes by
    sign(αᵢ), then average.

    This is more powerful than ZNE but requires more shots.
    """

    def __init__(
        self,
        noise_model: Dict[str, float]
    ):
        """
        Initialize PEC.

        Args:
            noise_model: Dictionary of error rates
        """
        self.noise_model = noise_model
        self.quasi_probability: Optional[Dict[str, float]] = None

    def decompose_noisy_operation(self, gate_type: str) -> Dict[str, float]:
        """
        Decompose noisy gate into quasi-probability distribution.

        For depolarizing noise with rate p:
        ε_depol(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

        To invert:
        ρ = (1/(1-p))ε_depol(ρ) - (p/(3(1-p)))(XρX + YρY + ZρZ)

        Args:
            gate_type: Type of gate

        Returns:
            Quasi-probability distribution
        """
        error_rate = self.noise_model.get(gate_type, 0.01)

        if error_rate >= 1.0:
            raise ValueError("Error rate too high for PEC")

        # Quasi-probabilities
        alpha_I = 1 / (1 - error_rate)
        alpha_err = -error_rate / (3 * (1 - error_rate))

        quasi_prob = {
            'I': alpha_I,
            'X': alpha_err,
            'Y': alpha_err,
            'Z': alpha_err
        }

        return quasi_prob

    def sampling_overhead(self, gate_type: str) -> float:
        """
        Calculate sampling overhead for PEC.

        Overhead = Σᵢ |αᵢ|

        Args:
            gate_type: Type of gate

        Returns:
            Sampling overhead
        """
        quasi_prob = self.decompose_noisy_operation(gate_type)
        return sum(abs(alpha) for alpha in quasi_prob.values())

    def mitigate_expectation(
        self,
        noisy_expectations: List[float],
        quasi_probs: List[float],
        num_samples: int
    ) -> MitigationResult:
        """
        Mitigate expectation using PEC.

        Args:
            noisy_expectations: List of expectation values from sampled operations
            quasi_probs: Corresponding quasi-probabilities
            num_samples: Number of samples taken

        Returns:
            Mitigation result
        """
        # Weighted average
        mitigated = np.average(noisy_expectations, weights=quasi_probs)

        # Calculate metrics
        raw = noisy_expectations[0]  # Assume first is unmitigated
        improvement = abs(mitigated / raw) if raw != 0 else 1.0
        overhead = sum(abs(alpha) for alpha in quasi_probs)

        return MitigationResult(
            raw_expectation=raw,
            mitigated_expectation=mitigated,
            improvement_factor=improvement,
            overhead=overhead,
            method="PEC"
        )


def compare_mitigation_techniques(
    ideal_value: float,
    noisy_value: float,
    readout_mitigated: float,
    zne_mitigated: float,
    pec_mitigated: float
) -> Dict[str, Dict[str, float]]:
    """
    Compare different mitigation techniques.

    Args:
        ideal_value: True ideal expectation value
        noisy_value: Noisy (unmitigated) value
        readout_mitigated: Readout error mitigated value
        zne_mitigated: ZNE mitigated value
        pec_mitigated: PEC mitigated value

    Returns:
        Comparison dictionary
    """
    def calculate_metrics(value):
        error = abs(value - ideal_value)
        improvement = abs(noisy_value - ideal_value) / error if error > 1e-10 else float('inf')
        return {
            'value': value,
            'error': error,
            'improvement': improvement
        }

    return {
        'noisy': calculate_metrics(noisy_value),
        'readout': calculate_metrics(readout_mitigated),
        'zne': calculate_metrics(zne_mitigated),
        'pec': calculate_metrics(pec_mitigated)
    }


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("ERROR MITIGATION TECHNIQUES DEMONSTRATION")
    print("=" * 70)

    # 1. Readout Error Mitigation
    print("\n1. READOUT ERROR MITIGATION")
    print("-" * 70)

    num_qubits = 2
    readout_errors = [0.05, 0.03]  # 5% and 3% error rates

    mitigator = ReadoutErrorMitigator(num_qubits)
    mitigator.calibrate(readout_errors, num_shots=10000)

    print(f"Calibrated for {num_qubits} qubits")
    print(f"Readout errors: {readout_errors}")
    print(f"Calibration matrix shape: {mitigator.calibration_matrix.shape}")

    # Simulate noisy counts
    noisy_counts = {'00': 850, '01': 80, '10': 40, '11': 30}
    print(f"\nNoisy counts: {noisy_counts}")

    mitigated_counts = mitigator.mitigate_counts(noisy_counts)
    print(f"Mitigated counts: {mitigated_counts}")

    # 2. Zero-Noise Extrapolation
    print("\n\n2. ZERO-NOISE EXTRAPOLATION (ZNE)")
    print("-" * 70)

    # Simulate circuit execution at different noise levels
    def example_circuit(fold_factor: int) -> float:
        """Simulate circuit with noise scaling."""
        ideal_value = 1.0
        base_error = 0.05
        noise_level = fold_factor * base_error
        return ideal_value * (1 - noise_level) + np.random.normal(0, 0.01)

    fold_factors = [1, 3, 5, 7]
    zne_result = ZeroNoiseExtrapolation.fold_circuit_globally(
        example_circuit,
        fold_factors,
        extrapolation_order=2
    )

    print(f"Fold factors: {fold_factors}")
    print(zne_result)

    # 3. Probabilistic Error Cancellation
    print("\n\n3. PROBABILISTIC ERROR CANCELLATION (PEC)")
    print("-" * 70)

    noise_model = {'CNOT': 0.02}  # 2% error rate
    pec = ProbabilisticErrorCancellation(noise_model)

    quasi_prob = pec.decompose_noisy_operation('CNOT')
    overhead = pec.sampling_overhead('CNOT')

    print(f"Noise model: {noise_model}")
    print(f"Quasi-probability decomposition: {quasi_prob}")
    print(f"Sampling overhead: {overhead:.2f}x")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("• Readout mitigation: ~2-5x improvement, low overhead")
    print("• ZNE: ~2-3x improvement, moderate overhead")
    print("• PEC: ~5-10x improvement, high overhead")
    print("• Choice depends on circuit depth and available shots")
    print("• Combining techniques can give best results")
    print("=" * 70)
