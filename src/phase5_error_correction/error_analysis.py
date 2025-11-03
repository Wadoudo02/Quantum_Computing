"""
Error Analysis and Threshold Calculations

This module provides tools for analyzing error correction performance,
calculating error thresholds, and understanding the overhead required
for fault-tolerant quantum computation.

Key concepts:
- Physical error rate: Error probability per gate/time step
- Logical error rate: Error probability of the encoded logical qubit
- Error threshold: Physical error rate below which logical error decreases
- Overhead: Number of physical qubits per logical qubit

Author: Quantum Computing Learning Project
Phase: 5 - Error Correction
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dataclasses import dataclass


@dataclass
class ErrorCorrectionPerformance:
    """
    Data class to store error correction performance metrics.

    Attributes:
        physical_error_rate: Physical error probability
        logical_error_rate: Logical error probability after correction
        overhead: Number of physical qubits per logical qubit
        syndrome_success_rate: Success rate of syndrome measurement
        recovery_success_rate: Success rate of recovery operation
        total_success_rate: Overall success rate
    """
    physical_error_rate: float
    logical_error_rate: float
    overhead: int
    syndrome_success_rate: float = 1.0
    recovery_success_rate: float = 1.0
    total_success_rate: float = 1.0


class ErrorAnalyzer:
    """
    Tools for analyzing quantum error correction performance.

    This class provides methods for:
    - Computing logical error rates from physical error rates
    - Analyzing error thresholds
    - Calculating resource overhead
    - Comparing different codes
    """

    def __init__(self):
        """Initialize the error analyzer."""
        pass

    @staticmethod
    def compute_uncorrected_error_rate(
        p_phys: float,
        n_qubits: int,
        n_gates: int
    ) -> float:
        """
        Compute error rate without error correction.

        For n qubits with n_gates applied, the probability of at least
        one error is approximately 1 - (1-p)^(n*n_gates).

        Args:
            p_phys: Physical error rate per gate
            n_qubits: Number of qubits
            n_gates: Number of gates

        Returns:
            float: Probability of at least one error
        """
        total_gates = n_qubits * n_gates
        return 1 - (1 - p_phys) ** total_gates

    @staticmethod
    def compute_three_qubit_logical_error(p_phys: float) -> float:
        """
        Compute logical error rate for 3-qubit bit-flip code.

        The code fails if 2 or 3 qubits have errors.
        P_logical = P(2 errors) + P(3 errors)
                  = 3p²(1-p) + p³
                  ≈ 3p² for small p

        Args:
            p_phys: Physical error rate

        Returns:
            float: Logical error rate
        """
        # Probability of 2 errors
        p_two = 3 * p_phys**2 * (1 - p_phys)

        # Probability of 3 errors
        p_three = p_phys**3

        return p_two + p_three

    @staticmethod
    def compute_shor_code_logical_error(p_phys: float) -> float:
        """
        Compute logical error rate for Shor's 9-qubit code.

        Simplified model: The code fails if more than one error occurs
        in any 3-qubit block, or if phase-flip detection fails.

        Args:
            p_phys: Physical error rate

        Returns:
            float: Logical error rate
        """
        # Probability that a single 3-qubit block fails (2+ errors)
        p_block_fail = 3 * p_phys**2 * (1 - p_phys) + p_phys**3

        # Probability that at least one of 3 blocks fails
        # Simplified: assume independent blocks
        p_any_block_fail = 1 - (1 - p_block_fail)**3

        return p_any_block_fail

    @staticmethod
    def compute_five_qubit_logical_error(p_phys: float) -> float:
        """
        Compute logical error rate for 5-qubit code.

        The code fails if 2 or more errors occur.
        P_logical = P(2+) = sum_{k=2}^5 C(5,k) p^k (1-p)^(5-k)

        Args:
            p_phys: Physical error rate

        Returns:
            float: Logical error rate
        """
        from math import comb

        p_logical = 0
        for k in range(2, 6):  # 2, 3, 4, 5 errors
            p_logical += comb(5, k) * p_phys**k * (1 - p_phys)**(5 - k)

        return p_logical

    @staticmethod
    def compute_surface_code_logical_error(
        p_phys: float,
        code_distance: int
    ) -> float:
        """
        Compute logical error rate for surface code (approximate).

        For surface codes, the logical error rate scales as:
        P_logical ≈ (p/p_th)^((d+1)/2)

        where p_th ≈ 0.01 is the threshold and d is the code distance.

        Args:
            p_phys: Physical error rate
            code_distance: Code distance d

        Returns:
            float: Logical error rate (approximate)
        """
        p_threshold = 0.01  # Typical surface code threshold

        if p_phys >= p_threshold:
            # Above threshold: error correction makes things worse
            return 1.0

        # Below threshold: exponential suppression
        return (p_phys / p_threshold) ** ((code_distance + 1) / 2)

    def analyze_threshold_behavior(
        self,
        error_rates: List[float],
        code_function: Callable[[float], float],
        code_name: str = "Code"
    ) -> Tuple[List[float], List[float], Optional[float]]:
        """
        Analyze error correction threshold behavior.

        Args:
            error_rates: List of physical error rates to test
            code_function: Function that computes logical error from physical
            code_name: Name of the code for display

        Returns:
            Tuple of (physical_rates, logical_rates, threshold)
        """
        logical_rates = [code_function(p) for p in error_rates]

        # Find approximate threshold (where logical = physical)
        threshold = None
        for i in range(len(error_rates) - 1):
            if logical_rates[i] < error_rates[i] and logical_rates[i+1] > error_rates[i+1]:
                # Threshold is between error_rates[i] and error_rates[i+1]
                threshold = (error_rates[i] + error_rates[i+1]) / 2
                break

        return error_rates, logical_rates, threshold

    def compute_overhead_analysis(
        self,
        codes: Dict[str, Tuple[int, int, Callable]]
    ) -> Dict[str, Dict]:
        """
        Analyze overhead for different codes.

        Args:
            codes: Dictionary of {code_name: (n_physical, n_logical, error_func)}

        Returns:
            Dictionary of overhead metrics per code
        """
        results = {}

        for name, (n_phys, n_log, error_func) in codes.items():
            overhead = n_phys / n_log

            # Calculate logical error at p=0.001 (typical target)
            p_logical = error_func(0.001)

            results[name] = {
                'physical_qubits': n_phys,
                'logical_qubits': n_log,
                'overhead': overhead,
                'logical_error_at_0.1%': p_logical,
                'improvement_factor': 0.001 / p_logical if p_logical > 0 else float('inf')
            }

        return results

    @staticmethod
    def estimate_gates_before_failure(
        p_error: float,
        failure_threshold: float = 0.5
    ) -> int:
        """
        Estimate number of gates before failure probability exceeds threshold.

        For n gates with error probability p per gate:
        P_failure ≈ 1 - (1-p)^n ≈ np for small p

        Args:
            p_error: Error probability per gate
            failure_threshold: Acceptable failure probability (default 0.5)

        Returns:
            int: Number of gates before failure threshold
        """
        if p_error <= 0:
            return int(1e10)  # Effectively infinite

        # Solve: 1 - (1-p)^n = threshold
        # n = log(1 - threshold) / log(1 - p)
        n_gates = np.log(1 - failure_threshold) / np.log(1 - p_error)

        return int(n_gates)

    @staticmethod
    def compute_concatenation_levels(
        p_phys: float,
        p_target: float,
        error_suppression_factor: float = 100
    ) -> int:
        """
        Compute number of concatenation levels needed.

        Each level of concatenation suppresses errors by a factor roughly
        proportional to the square of the error rate.

        Args:
            p_phys: Physical error rate
            p_target: Target logical error rate
            error_suppression_factor: Error suppression per level

        Returns:
            int: Number of concatenation levels needed
        """
        if p_phys <= p_target:
            return 0

        # Each level: p → (p/p_th)^k where k ~ 2-3
        levels = 0
        p_current = p_phys

        while p_current > p_target and levels < 10:  # Safety limit
            p_current = p_current**2 * error_suppression_factor
            levels += 1

        return levels


class ThresholdCalculator:
    """
    Calculator for error correction thresholds.

    The threshold is the physical error rate below which adding more
    error correction improves the logical error rate.
    """

    def __init__(self):
        """Initialize threshold calculator."""
        self.analyzer = ErrorAnalyzer()

    def find_threshold(
        self,
        code_function: Callable[[float], float],
        p_range: Tuple[float, float] = (1e-5, 0.2),
        tolerance: float = 1e-4
    ) -> float:
        """
        Find the error threshold using binary search.

        The threshold is where P_logical ≈ P_physical.

        Args:
            code_function: Function computing logical error from physical
            p_range: Range of physical error rates to search
            tolerance: Convergence tolerance

        Returns:
            float: Threshold error rate
        """
        p_low, p_high = p_range

        while p_high - p_low > tolerance:
            p_mid = (p_low + p_high) / 2
            p_logical = code_function(p_mid)

            # At threshold: p_logical ≈ p_physical
            if p_logical < p_mid:
                # Below threshold: error correction helps
                p_low = p_mid
            else:
                # Above threshold: error correction hurts
                p_high = p_mid

        return (p_low + p_high) / 2

    def compare_codes(
        self,
        error_rates: np.ndarray,
        codes: Dict[str, Callable[[float], float]]
    ) -> Dict[str, np.ndarray]:
        """
        Compare multiple codes across error rates.

        Args:
            error_rates: Array of physical error rates
            codes: Dictionary of {name: error_function}

        Returns:
            Dictionary of {code_name: logical_error_rates}
        """
        results = {}

        for name, code_func in codes.items():
            logical_rates = np.array([code_func(p) for p in error_rates])
            results[name] = logical_rates

        return results


def demonstrate_error_analysis():
    """Demonstrate error analysis tools."""
    print("=" * 70)
    print("ERROR ANALYSIS AND THRESHOLDS")
    print("=" * 70)

    analyzer = ErrorAnalyzer()

    print("\n1. UNCORRECTED ERROR ACCUMULATION")
    print("-" * 70)

    p_phys = 0.001  # 0.1% error rate
    for n_gates in [10, 100, 1000, 10000]:
        p_error = analyzer.compute_uncorrected_error_rate(p_phys, 1, n_gates)
        n_safe = analyzer.estimate_gates_before_failure(p_phys)
        print(f"After {n_gates:5d} gates: {p_error:6.2%} failure probability")

    print(f"\nGates before 50% failure at p={p_phys}: {n_safe:,}")

    print("\n2. LOGICAL ERROR RATES")
    print("-" * 70)

    test_rates = [0.001, 0.01, 0.05, 0.1]

    print("\nPhysical | 3-Qubit  | Shor Code | 5-Qubit  | Improvement")
    print("-" * 70)

    for p in test_rates:
        p_3qubit = analyzer.compute_three_qubit_logical_error(p)
        p_shor = analyzer.compute_shor_code_logical_error(p)
        p_5qubit = analyzer.compute_five_qubit_logical_error(p)

        improvement = p / p_5qubit if p_5qubit > 0 else float('inf')

        print(f"{p:7.2%}  | {p_3qubit:7.2%}  | {p_shor:8.2%}  | "
              f"{p_5qubit:7.2%}  | {improvement:5.1f}x")

    print("\n3. ERROR THRESHOLD ANALYSIS")
    print("-" * 70)

    calculator = ThresholdCalculator()

    # Find thresholds for different codes
    threshold_3qubit = calculator.find_threshold(
        analyzer.compute_three_qubit_logical_error
    )

    print(f"3-Qubit code threshold: {threshold_3qubit:.4f} ({threshold_3qubit:.2%})")
    print(f"Note: 3-qubit code has no true threshold (logical > physical)")

    print("\n4. OVERHEAD ANALYSIS")
    print("-" * 70)

    codes = {
        "3-Qubit": (3, 1, analyzer.compute_three_qubit_logical_error),
        "Shor Code": (9, 1, analyzer.compute_shor_code_logical_error),
        "5-Qubit": (5, 1, analyzer.compute_five_qubit_logical_error),
    }

    overhead = analyzer.compute_overhead_analysis(codes)

    print(f"\n{'Code':<15} {'Physical':<10} {'Logical':<10} {'Overhead':<10} "
          f"{'P_L @ 0.1%':<12}")
    print("-" * 70)

    for name, metrics in overhead.items():
        print(f"{name:<15} {metrics['physical_qubits']:<10} "
              f"{metrics['logical_qubits']:<10} {metrics['overhead']:<10.1f} "
              f"{metrics['logical_error_at_0.1%']:<12.2e}")

    print("\n5. CONCATENATION REQUIREMENTS")
    print("-" * 70)

    p_phys = 0.001
    target_rates = [1e-6, 1e-9, 1e-12, 1e-15]

    print(f"\nPhysical error rate: {p_phys:.3f}")
    print(f"\nTarget P_L   | Levels Needed | Total Physical Qubits (5-qubit)")
    print("-" * 70)

    for p_target in target_rates:
        levels = analyzer.compute_concatenation_levels(p_phys, p_target)
        # Each level increases qubits by factor of 5
        total_qubits = 5 ** levels

        print(f"{p_target:<12.0e} | {levels:<13} | {total_qubits:,}")

    print("\n6. GATES BEFORE FAILURE")
    print("-" * 70)

    print(f"\n{'Error Rate':<15} {'Gates (50% fail)':<20} {'Gates (1% fail)'}")
    print("-" * 70)

    for p in [0.1, 0.01, 0.001, 0.0001]:
        gates_50 = analyzer.estimate_gates_before_failure(p, 0.5)
        gates_1 = analyzer.estimate_gates_before_failure(p, 0.01)
        print(f"{p:<15.4f} {gates_50:<20,} {gates_1:,}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demonstrate_error_analysis()
