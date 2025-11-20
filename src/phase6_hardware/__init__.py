"""
Phase 6: Real Quantum Hardware & NISQ Computing

This phase bridges theory and practice by implementing:
- Hardware interfaces for IBM, IonQ, Rigetti
- Realistic noise models based on actual hardware specs
- Circuit transpilation and optimization
- Error mitigation techniques (ZNE, PEC, readout correction)
- NISQ algorithms (VQE, QAOA, quantum teleportation)
- Hardware benchmarking and characterization tools
- Analysis and visualization utilities

Author: Wadoud Charbak
Date: November 2024
For: Quantinuum & Riverlane Recruitment
"""

from .hardware_interface import (
    HardwareSpecs,
    BackendType,
    QuantumBackend,
    get_backend_specs,
    compare_backends,
    IBM_JAKARTA_SPECS,
    IBM_SIMULATOR_SPECS,
    IONQ_HARMONY_SPECS,
    IONQ_SIMULATOR_SPECS,
    RIGETTI_ASPEN_SPECS,
    RIGETTI_SIMULATOR_SPECS
)

from .noise_models import (
    NoiseParameters,
    RealisticNoiseModel,
    create_noise_model
)

from .transpiler import (
    GateType,
    Gate,
    QuantumCircuit,
    CircuitTranspiler
)

from .error_mitigation import (
    MitigationResult,
    ReadoutErrorMitigator,
    ZeroNoiseExtrapolation,
    ProbabilisticErrorCancellation,
    compare_mitigation_techniques
)

from .benchmarking import (
    BenchmarkResult,
    RandomizedBenchmarking,
    QuantumVolume,
    CoherenceTimeMeasurement
)

from .nisq_algorithms import (
    VQEResult,
    VariationalQuantumEigensolver,
    QAOAResult,
    QuantumApproximateOptimizationAlgorithm,
    QuantumTeleportation
)

from .analysis_tools import (
    plot_backend_comparison,
    plot_circuit_fidelity_vs_depth,
    plot_error_mitigation_comparison,
    plot_connectivity_graph,
    plot_decoherence_curves,
    create_benchmark_summary_table
)

__all__ = [
    # Hardware interface
    'HardwareSpecs',
    'BackendType',
    'QuantumBackend',
    'get_backend_specs',
    'compare_backends',
    'IBM_JAKARTA_SPECS',
    'IBM_SIMULATOR_SPECS',
    'IONQ_HARMONY_SPECS',
    'IONQ_SIMULATOR_SPECS',
    'RIGETTI_ASPEN_SPECS',
    'RIGETTI_SIMULATOR_SPECS',

    # Noise models
    'NoiseParameters',
    'RealisticNoiseModel',
    'create_noise_model',

    # Transpiler
    'GateType',
    'Gate',
    'QuantumCircuit',
    'CircuitTranspiler',

    # Error mitigation
    'MitigationResult',
    'ReadoutErrorMitigator',
    'ZeroNoiseExtrapolation',
    'ProbabilisticErrorCancellation',
    'compare_mitigation_techniques',

    # Benchmarking
    'BenchmarkResult',
    'RandomizedBenchmarking',
    'QuantumVolume',
    'CoherenceTimeMeasurement',

    # NISQ algorithms
    'VQEResult',
    'VariationalQuantumEigensolver',
    'QAOAResult',
    'QuantumApproximateOptimizationAlgorithm',
    'QuantumTeleportation',

    # Analysis tools
    'plot_backend_comparison',
    'plot_circuit_fidelity_vs_depth',
    'plot_error_mitigation_comparison',
    'plot_connectivity_graph',
    'plot_decoherence_curves',
    'create_benchmark_summary_table',
]

__version__ = '1.0.0'
__author__ = 'Wadoud Charbak'
