"""
Noise Models Module for Phase 6: Realistic Quantum Noise Simulation

This module provides realistic noise models based on actual quantum hardware
specifications. It builds on Phase 4's noise channels to create complete
hardware noise models for IBM, IonQ, and Rigetti systems.

Author: Wadoud Charbak
Date: November 2024
For: Quantinuum & Riverlane Recruitment
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path to import from other phases
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase4_noise.density_matrix import DensityMatrix
from phase4_noise.quantum_channels import (
    apply_depolarizing_channel,
    apply_amplitude_damping,
    apply_phase_damping
)
from phase6_hardware.hardware_interface import HardwareSpecs, BackendType


@dataclass
class NoiseParameters:
    """
    Consolidated noise parameters for a quantum system.

    Attributes:
        single_qubit_error: Single-qubit gate error rate
        two_qubit_error: Two-qubit gate error rate
        readout_error: Measurement error rate
        t1: T1 (energy relaxation) time in μs
        t2: T2 (dephasing) time in μs
        gate_time_1q: Single-qubit gate time in ns
        gate_time_2q: Two-qubit gate time in ns
    """
    single_qubit_error: float
    two_qubit_error: float
    readout_error: float
    t1: float  # microseconds
    t2: float  # microseconds
    gate_time_1q: float = 50.0  # nanoseconds
    gate_time_2q: float = 300.0  # nanoseconds

    def __post_init__(self):
        """Validate noise parameters."""
        assert 0 <= self.single_qubit_error <= 1, "Error rates must be in [0, 1]"
        assert 0 <= self.two_qubit_error <= 1, "Error rates must be in [0, 1]"
        assert 0 <= self.readout_error <= 1, "Error rates must be in [0, 1]"
        assert self.t1 > 0, "T1 must be positive"
        assert self.t2 > 0, "T2 must be positive"
        assert self.t2 <= 2 * self.t1, "T2 must satisfy T2 <= 2*T1"

    def depolarizing_param_1q(self) -> float:
        """
        Calculate depolarizing parameter for single-qubit gates.

        The depolarizing parameter p relates to gate fidelity F via:
        F = 1 - (3/4)p  =>  p = (4/3)(1 - F)
        """
        return min((4/3) * self.single_qubit_error, 1.0)

    def depolarizing_param_2q(self) -> float:
        """Calculate depolarizing parameter for two-qubit gates."""
        return min((4/3) * self.two_qubit_error, 1.0)

    def damping_param_1q(self) -> float:
        """
        Calculate amplitude damping parameter for single-qubit gate.

        γ = 1 - exp(-t/T1)
        """
        time_ratio = (self.gate_time_1q * 1e-3) / self.t1  # Convert ns to μs
        return 1.0 - np.exp(-time_ratio)

    def damping_param_2q(self) -> float:
        """Calculate amplitude damping parameter for two-qubit gate."""
        time_ratio = (self.gate_time_2q * 1e-3) / self.t1
        return 1.0 - np.exp(-time_ratio)

    def dephasing_param_1q(self) -> float:
        """
        Calculate pure dephasing parameter for single-qubit gate.

        For pure dephasing: γ_φ = 1/T_φ where 1/T2 = 1/(2T1) + 1/T_φ
        Therefore: T_φ = 1 / (1/T2 - 1/(2T1))
        Then: λ = 1 - exp(-t/T_φ)
        """
        if self.t2 >= 2 * self.t1:
            return 0.0  # T2-limited regime

        t_phi = 1.0 / (1.0/self.t2 - 1.0/(2.0*self.t1))
        time_ratio = (self.gate_time_1q * 1e-3) / t_phi
        return 1.0 - np.exp(-time_ratio)

    def dephasing_param_2q(self) -> float:
        """Calculate pure dephasing parameter for two-qubit gate."""
        if self.t2 >= 2 * self.t1:
            return 0.0

        t_phi = 1.0 / (1.0/self.t2 - 1.0/(2.0*self.t1))
        time_ratio = (self.gate_time_2q * 1e-3) / t_phi
        return 1.0 - np.exp(-time_ratio)


class RealisticNoiseModel:
    """
    Realistic noise model based on actual hardware specifications.

    This combines multiple noise sources:
    1. Depolarizing noise (gate imperfections)
    2. Amplitude damping (T1 decay)
    3. Phase damping (T2 dephasing)
    4. Readout errors
    """

    def __init__(self, hardware_specs: HardwareSpecs):
        """
        Initialize noise model from hardware specs.

        Args:
            hardware_specs: Hardware specifications
        """
        self.specs = hardware_specs
        self.noise_params = self._extract_noise_parameters()

    def _extract_noise_parameters(self) -> List[NoiseParameters]:
        """
        Extract noise parameters for each qubit.

        Returns:
            List of NoiseParameters, one per qubit
        """
        params_list = []

        for i in range(self.specs.num_qubits):
            # Single-qubit error from gate fidelity
            sq_fidelity = self.specs.gate_fidelities.get('single_qubit', 1.0)
            sq_error = 1.0 - sq_fidelity

            # Two-qubit error
            tq_fidelity = self.specs.gate_fidelities.get('cnot', 1.0)
            tq_error = 1.0 - tq_fidelity

            # Readout error
            ro_fidelity = self.specs.readout_fidelity
            ro_error = 1.0 - ro_fidelity

            # Decoherence times
            t1 = float(self.specs.t1_times[i]) if i < len(self.specs.t1_times) else 100.0
            t2 = float(self.specs.t2_times[i]) if i < len(self.specs.t2_times) else 50.0

            # Gate times depend on backend type
            if self.specs.backend_type in [BackendType.IBM_HARDWARE, BackendType.IBM_SIMULATOR]:
                gate_time_1q = 50.0  # ns
                gate_time_2q = 300.0  # ns
            elif self.specs.backend_type in [BackendType.IONQ_HARDWARE, BackendType.IONQ_SIMULATOR]:
                gate_time_1q = 10.0  # Faster for trapped ions
                gate_time_2q = 500.0  # MS gate is slower
            else:  # Rigetti
                gate_time_1q = 40.0
                gate_time_2q = 200.0

            params = NoiseParameters(
                single_qubit_error=sq_error,
                two_qubit_error=tq_error,
                readout_error=ro_error,
                t1=t1,
                t2=t2,
                gate_time_1q=gate_time_1q,
                gate_time_2q=gate_time_2q
            )

            params_list.append(params)

        return params_list

    def apply_single_qubit_noise(
        self,
        rho: DensityMatrix,
        qubit_idx: int
    ) -> DensityMatrix:
        """
        Apply realistic single-qubit gate noise.

        Combines depolarizing, amplitude damping, and phase damping.

        Args:
            rho: Input density matrix
            qubit_idx: Index of qubit

        Returns:
            Noisy density matrix
        """
        if qubit_idx >= len(self.noise_params):
            return rho  # No noise if qubit not in specs

        params = self.noise_params[qubit_idx]

        # Apply depolarizing noise
        p_depol = params.depolarizing_param_1q()
        rho_noisy = apply_depolarizing_channel(rho, p_depol)

        # Apply amplitude damping
        gamma_amp = params.damping_param_1q()
        rho_noisy = apply_amplitude_damping(rho_noisy, gamma_amp)

        # Apply phase damping
        lambda_phase = params.dephasing_param_1q()
        rho_noisy = apply_phase_damping(rho_noisy, lambda_phase)

        return rho_noisy

    def apply_two_qubit_noise(
        self,
        rho: DensityMatrix,
        qubit_indices: Tuple[int, int]
    ) -> DensityMatrix:
        """
        Apply realistic two-qubit gate noise.

        Args:
            rho: Input density matrix
            qubit_indices: Tuple of qubit indices

        Returns:
            Noisy density matrix
        """
        # Use average noise parameters from both qubits
        params_list = [self.noise_params[i] for i in qubit_indices
                      if i < len(self.noise_params)]

        if not params_list:
            return rho

        avg_tq_error = np.mean([p.two_qubit_error for p in params_list])
        avg_t1 = np.mean([p.t1 for p in params_list])
        avg_t2 = np.mean([p.t2 for p in params_list])
        avg_gate_time = params_list[0].gate_time_2q

        # Calculate noise parameters
        p_depol = min((4/3) * avg_tq_error, 1.0)

        time_ratio_t1 = (avg_gate_time * 1e-3) / avg_t1
        gamma_amp = 1.0 - np.exp(-time_ratio_t1)

        # Phase damping
        if avg_t2 < 2 * avg_t1:
            t_phi = 1.0 / (1.0/avg_t2 - 1.0/(2.0*avg_t1))
            time_ratio_phi = (avg_gate_time * 1e-3) / t_phi
            lambda_phase = 1.0 - np.exp(-time_ratio_phi)
        else:
            lambda_phase = 0.0

        # Apply noise channels
        rho_noisy = apply_depolarizing_channel(rho, p_depol)

        # Apply damping to each qubit separately
        for qubit_idx in qubit_indices:
            if qubit_idx < self.specs.num_qubits:
                rho_noisy = apply_amplitude_damping(rho_noisy, gamma_amp)
                rho_noisy = apply_phase_damping(rho_noisy, lambda_phase)

        return rho_noisy

    def apply_readout_noise(
        self,
        counts: Dict[str, int],
        num_qubits: int
    ) -> Dict[str, int]:
        """
        Apply readout (measurement) noise to measurement counts.

        Models classical bit-flip errors during measurement.

        Args:
            counts: Ideal measurement counts
            num_qubits: Number of qubits

        Returns:
            Noisy measurement counts
        """
        # Average readout error across all qubits
        avg_readout_error = np.mean([p.readout_error for p in self.noise_params[:num_qubits]])

        if avg_readout_error < 1e-10:
            return counts  # No readout noise

        # Create confusion matrix for readout
        # P(measure i | prepared j)
        total_shots = sum(counts.values())
        noisy_counts = {state: 0 for state in counts.keys()}

        for state, count in counts.items():
            # For each measured state, distribute counts according to readout errors
            state_bits = list(state)

            for _ in range(count):
                # Flip each bit with probability avg_readout_error
                noisy_state_bits = []
                for bit in state_bits:
                    if np.random.rand() < avg_readout_error:
                        # Flip bit
                        noisy_bit = '1' if bit == '0' else '0'
                        noisy_state_bits.append(noisy_bit)
                    else:
                        noisy_state_bits.append(bit)

                noisy_state = ''.join(noisy_state_bits)
                if noisy_state not in noisy_counts:
                    noisy_counts[noisy_state] = 0
                noisy_counts[noisy_state] += 1

        # Remove zero counts
        noisy_counts = {state: count for state, count in noisy_counts.items() if count > 0}

        return noisy_counts

    def get_effective_error_rate(self, gate_type: str = 'single') -> float:
        """
        Get effective error rate for gate type.

        Args:
            gate_type: 'single' or 'two'

        Returns:
            Average error rate
        """
        if gate_type == 'single':
            return np.mean([p.single_qubit_error for p in self.noise_params])
        elif gate_type == 'two':
            return np.mean([p.two_qubit_error for p in self.noise_params])
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

    def estimate_circuit_fidelity(
        self,
        num_single_qubit_gates: int,
        num_two_qubit_gates: int
    ) -> float:
        """
        Estimate overall circuit fidelity.

        Assumes independent errors:
        F_circuit ≈ (1 - p_1q)^n_1q * (1 - p_2q)^n_2q

        Args:
            num_single_qubit_gates: Number of single-qubit gates
            num_two_qubit_gates: Number of two-qubit gates

        Returns:
            Estimated circuit fidelity
        """
        p_1q = self.get_effective_error_rate('single')
        p_2q = self.get_effective_error_rate('two')

        fidelity = ((1 - p_1q) ** num_single_qubit_gates *
                   (1 - p_2q) ** num_two_qubit_gates)

        return fidelity

    def characterization_summary(self) -> Dict[str, float]:
        """
        Get summary of noise characteristics.

        Returns:
            Dictionary with noise metrics
        """
        return {
            'avg_single_qubit_error': self.get_effective_error_rate('single'),
            'avg_two_qubit_error': self.get_effective_error_rate('two'),
            'avg_readout_error': np.mean([p.readout_error for p in self.noise_params]),
            'avg_t1_us': np.mean([p.t1 for p in self.noise_params]),
            'avg_t2_us': np.mean([p.t2 for p in self.noise_params]),
            't2_t1_ratio': np.mean([p.t2/p.t1 for p in self.noise_params]),
        }


def create_noise_model(backend_name: str) -> RealisticNoiseModel:
    """
    Create noise model for a named backend.

    Args:
        backend_name: Name of backend

    Returns:
        Realistic noise model
    """
    from phase6_hardware.hardware_interface import get_backend_specs

    specs = get_backend_specs(backend_name)
    return RealisticNoiseModel(specs)


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("REALISTIC NOISE MODEL CHARACTERIZATION")
    print("=" * 70)

    backends = ['ibm_jakarta', 'ionq_harmony', 'rigetti_aspen_m3']

    for backend in backends:
        print(f"\n{backend.upper()}:")
        print("-" * 70)

        noise_model = create_noise_model(backend)
        summary = noise_model.characterization_summary()

        print(f"  Single-qubit error rate: {summary['avg_single_qubit_error']:.6f}")
        print(f"  Two-qubit error rate:    {summary['avg_two_qubit_error']:.6f}")
        print(f"  Readout error rate:      {summary['avg_readout_error']:.6f}")
        print(f"  Average T1:              {summary['avg_t1_us']:.2f} μs")
        print(f"  Average T2:              {summary['avg_t2_us']:.2f} μs")
        print(f"  T2/T1 ratio:             {summary['t2_t1_ratio']:.3f}")

        # Example circuit fidelity
        fidelity_10 = noise_model.estimate_circuit_fidelity(10, 5)
        fidelity_100 = noise_model.estimate_circuit_fidelity(100, 50)

        print(f"\n  Estimated Circuit Fidelity:")
        print(f"    10 single-qubit + 5 two-qubit gates:  {fidelity_10:.4f}")
        print(f"    100 single-qubit + 50 two-qubit gates: {fidelity_100:.4f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("• Trapped ions (IonQ) have lower error rates")
    print("• Superconducting qubits have faster gate times but higher errors")
    print("• Circuit fidelity degrades exponentially with depth")
    print("• Error correction is essential for deep circuits")
    print("=" * 70)
