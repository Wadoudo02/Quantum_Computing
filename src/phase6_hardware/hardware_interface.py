"""
Hardware Interface Module for Phase 6: Real Quantum Hardware & NISQ Computing

This module provides an abstract interface for interacting with various quantum
hardware backends (IBM Quantum, IonQ, Rigetti) and their simulators.

Author: Wadoud Charbak
Date: November 2024
For: Quantinuum & Riverlane Recruitment
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum


class BackendType(Enum):
    """Enum for different quantum backend types."""
    IBM_SIMULATOR = "ibm_simulator"
    IBM_HARDWARE = "ibm_hardware"
    IONQ_SIMULATOR = "ionq_simulator"
    IONQ_HARDWARE = "ionq_hardware"
    RIGETTI_SIMULATOR = "rigetti_simulator"
    RIGETTI_HARDWARE = "rigetti_hardware"
    IDEAL = "ideal"


@dataclass
class HardwareSpecs:
    """
    Hardware specifications for quantum backends.

    Attributes:
        name: Backend name
        num_qubits: Number of qubits available
        connectivity: List of qubit pairs that can interact
        t1_times: T1 (energy relaxation) times in microseconds
        t2_times: T2 (dephasing) times in microseconds
        gate_fidelities: Dictionary of gate fidelities
        readout_fidelity: Measurement fidelity
        backend_type: Type of backend
    """
    name: str
    num_qubits: int
    connectivity: List[Tuple[int, int]]
    t1_times: np.ndarray  # microseconds
    t2_times: np.ndarray  # microseconds
    gate_fidelities: Dict[str, float]
    readout_fidelity: float
    backend_type: BackendType

    def __post_init__(self):
        """Validate hardware specs."""
        # Ensure T2 <= 2*T1 (physical constraint)
        if len(self.t1_times) > 0 and len(self.t2_times) > 0:
            assert np.all(self.t2_times <= 2 * self.t1_times), \
                "T2 must be <= 2*T1 (physical constraint)"

    def average_gate_fidelity(self) -> float:
        """Calculate average gate fidelity."""
        if not self.gate_fidelities:
            return 1.0
        return np.mean(list(self.gate_fidelities.values()))

    def connectivity_degree(self) -> float:
        """
        Calculate connectivity as fraction of all possible connections.
        Returns value between 0 (no connectivity) and 1 (fully connected).
        """
        if self.num_qubits <= 1:
            return 1.0
        max_connections = self.num_qubits * (self.num_qubits - 1) / 2
        actual_connections = len(self.connectivity)
        return actual_connections / max_connections


class QuantumBackend(ABC):
    """
    Abstract base class for quantum backends.

    This provides a unified interface for running quantum circuits on
    different hardware platforms (IBM, IonQ, Rigetti) and their simulators.
    """

    def __init__(self, specs: HardwareSpecs):
        """
        Initialize quantum backend.

        Args:
            specs: Hardware specifications
        """
        self.specs = specs
        self._circuit_cache: Dict[str, Any] = {}

    @abstractmethod
    def run_circuit(
        self,
        circuit: Any,
        shots: int = 1024,
        **kwargs
    ) -> Dict[str, int]:
        """
        Run a quantum circuit and return measurement results.

        Args:
            circuit: Quantum circuit to run
            shots: Number of measurement shots
            **kwargs: Additional backend-specific parameters

        Returns:
            Dictionary mapping measurement outcomes to counts
        """
        pass

    @abstractmethod
    def transpile_circuit(
        self,
        circuit: Any,
        optimization_level: int = 2
    ) -> Any:
        """
        Transpile circuit for hardware constraints.

        Args:
            circuit: Input circuit
            optimization_level: Optimization level (0-3)

        Returns:
            Transpiled circuit
        """
        pass

    def get_specs(self) -> HardwareSpecs:
        """Get hardware specifications."""
        return self.specs

    def estimate_runtime(self, circuit_depth: int, shots: int = 1024) -> float:
        """
        Estimate circuit runtime in seconds.

        Args:
            circuit_depth: Circuit depth (number of layers)
            shots: Number of shots

        Returns:
            Estimated runtime in seconds
        """
        # Rough estimate: gate time + readout time
        gate_time = 100e-9  # 100 ns per gate (typical)
        readout_time = 1e-6  # 1 μs readout (typical)

        single_shot_time = circuit_depth * gate_time + readout_time
        total_time = single_shot_time * shots

        return total_time

    def check_connectivity(self, qubit_pairs: List[Tuple[int, int]]) -> bool:
        """
        Check if all qubit pairs in circuit are connected.

        Args:
            qubit_pairs: List of qubit pairs used in circuit

        Returns:
            True if all pairs are connected
        """
        for pair in qubit_pairs:
            if pair not in self.specs.connectivity and \
               (pair[1], pair[0]) not in self.specs.connectivity:
                return False
        return True


# Predefined hardware specifications based on real systems

IBM_JAKARTA_SPECS = HardwareSpecs(
    name="ibm_jakarta",
    num_qubits=7,
    connectivity=[(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (4, 6)],
    t1_times=np.array([142.3, 138.7, 135.2, 129.8, 145.6, 133.4, 140.1]),  # μs
    t2_times=np.array([95.4, 88.2, 91.7, 85.3, 97.8, 89.1, 93.6]),  # μs
    gate_fidelities={
        'single_qubit': 0.9995,
        'cnot': 0.987,
        'measurement': 0.952
    },
    readout_fidelity=0.952,
    backend_type=BackendType.IBM_HARDWARE
)

IBM_SIMULATOR_SPECS = HardwareSpecs(
    name="ibm_simulator",
    num_qubits=32,
    connectivity=[(i, j) for i in range(32) for j in range(i+1, 32)],  # Fully connected
    t1_times=np.ones(32) * 1e6,  # Very long (ideal)
    t2_times=np.ones(32) * 1e6,  # Very long (ideal)
    gate_fidelities={
        'single_qubit': 1.0,
        'cnot': 1.0,
        'measurement': 1.0
    },
    readout_fidelity=1.0,
    backend_type=BackendType.IBM_SIMULATOR
)

IONQ_HARMONY_SPECS = HardwareSpecs(
    name="ionq_harmony",
    num_qubits=11,
    connectivity=[(i, j) for i in range(11) for j in range(i+1, 11)],  # Fully connected
    t1_times=np.ones(11) * 1e6,  # Very long for trapped ions
    t2_times=np.ones(11) * 5e5,  # Limited by motional heating
    gate_fidelities={
        'single_qubit': 0.9998,
        'cnot': 0.972,  # MS gate
        'measurement': 0.996
    },
    readout_fidelity=0.996,
    backend_type=BackendType.IONQ_HARDWARE
)

IONQ_SIMULATOR_SPECS = HardwareSpecs(
    name="ionq_simulator",
    num_qubits=29,
    connectivity=[(i, j) for i in range(29) for j in range(i+1, 29)],  # Fully connected
    t1_times=np.ones(29) * 1e9,
    t2_times=np.ones(29) * 1e9,
    gate_fidelities={
        'single_qubit': 1.0,
        'cnot': 1.0,
        'measurement': 1.0
    },
    readout_fidelity=1.0,
    backend_type=BackendType.IONQ_SIMULATOR
)

RIGETTI_ASPEN_SPECS = HardwareSpecs(
    name="rigetti_aspen_m3",
    num_qubits=79,
    connectivity=[
        # Simplified connectivity (real Aspen-M-3 has complex topology)
        (i, i+1) for i in range(78)
    ] + [(i, i+8) for i in range(0, 71, 8)],
    t1_times=np.random.uniform(15, 35, 79),  # 15-35 μs typical
    t2_times=np.random.uniform(10, 25, 79),  # 10-25 μs typical
    gate_fidelities={
        'single_qubit': 0.998,
        'cnot': 0.90,  # CZ gate
        'measurement': 0.91
    },
    readout_fidelity=0.91,
    backend_type=BackendType.RIGETTI_HARDWARE
)

RIGETTI_SIMULATOR_SPECS = HardwareSpecs(
    name="rigetti_simulator",
    num_qubits=20,
    connectivity=[(i, j) for i in range(20) for j in range(i+1, 20)],
    t1_times=np.ones(20) * 1e6,
    t2_times=np.ones(20) * 1e6,
    gate_fidelities={
        'single_qubit': 1.0,
        'cnot': 1.0,
        'measurement': 1.0
    },
    readout_fidelity=1.0,
    backend_type=BackendType.RIGETTI_SIMULATOR
)


def get_backend_specs(backend_name: str) -> HardwareSpecs:
    """
    Get hardware specifications for a named backend.

    Args:
        backend_name: Name of backend

    Returns:
        Hardware specifications

    Raises:
        ValueError: If backend name not recognized
    """
    backends = {
        'ibm_jakarta': IBM_JAKARTA_SPECS,
        'ibm_simulator': IBM_SIMULATOR_SPECS,
        'ionq_harmony': IONQ_HARMONY_SPECS,
        'ionq_simulator': IONQ_SIMULATOR_SPECS,
        'rigetti_aspen_m3': RIGETTI_ASPEN_SPECS,
        'rigetti_simulator': RIGETTI_SIMULATOR_SPECS
    }

    if backend_name not in backends:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Available: {list(backends.keys())}"
        )

    return backends[backend_name]


def compare_backends(backend_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compare specifications across multiple backends.

    Args:
        backend_names: List of backend names to compare

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}

    for name in backend_names:
        specs = get_backend_specs(name)
        comparison[name] = {
            'num_qubits': specs.num_qubits,
            'avg_t1_us': float(np.mean(specs.t1_times)),
            'avg_t2_us': float(np.mean(specs.t2_times)),
            'avg_gate_fidelity': specs.average_gate_fidelity(),
            'readout_fidelity': specs.readout_fidelity,
            'connectivity': specs.connectivity_degree()
        }

    return comparison


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("QUANTUM HARDWARE SPECIFICATIONS COMPARISON")
    print("=" * 70)

    backends = [
        'ibm_jakarta',
        'ionq_harmony',
        'rigetti_aspen_m3'
    ]

    comparison = compare_backends(backends)

    print("\nHardware Comparison:")
    print("-" * 70)
    for backend, metrics in comparison.items():
        print(f"\n{backend.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                if metric == 'connectivity':
                    print(f"  {metric}: {value:.1%}")
                else:
                    print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("• IBM (Superconducting): Good coherence times, limited connectivity")
    print("• IonQ (Trapped Ions): Excellent fidelity, full connectivity")
    print("• Rigetti (Superconducting): Many qubits, lower fidelities")
    print("=" * 70)
