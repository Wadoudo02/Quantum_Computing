"""
Phase 3: Quantum Algorithms

This module implements foundational quantum algorithms demonstrating quantum advantage:
- Deutsch-Jozsa algorithm (exponential speedup)
- Grover's search algorithm (quadratic speedup)
- Quantum Fourier Transform (exponential speedup)

Modules:
--------
- gates: Multi-qubit quantum gates (Toffoli, CZ, SWAP, etc.)
- oracles: Oracle construction utilities for algorithm testing
- deutsch_jozsa: Deutsch-Jozsa algorithm implementation
- grover: Grover's search with amplitude amplification
- qft: Quantum Fourier Transform
- circuit_visualization: Circuit diagram generation
- performance_analysis: Classical vs quantum comparison tools
- app: Interactive Streamlit application

Author: Wadoud Charbak
Based on: Imperial College London Quantum Information Theory
For: Quantinuum & Riverlane recruitment
"""

# Core modules - gates and oracles
try:
    from .gates import (
        ControlledGate,
        controlled_z,
        controlled_u,
        toffoli,
        swap_gate,
        multi_controlled_gate,
        controlled_phase,
    )
except ImportError:
    pass

try:
    from .oracles import (
        Oracle,
        deutsch_jozsa_oracle,
        grover_oracle,
        create_balanced_function,
        create_constant_function,
    )
except ImportError:
    pass

# Algorithm modules
try:
    from .deutsch_jozsa import (
        deutsch_jozsa_algorithm,
        create_dj_circuit,
        verify_function_type,
    )
except ImportError:
    pass

try:
    from .grover import (
        grover_search,
        grover_diffusion,
        optimal_grover_iterations,
        amplitude_amplification_step,
    )
except ImportError:
    pass

try:
    from .qft import (
        quantum_fourier_transform,
        inverse_qft,
        qft_circuit,
        controlled_phase_gate,
    )
except ImportError:
    pass

# Visualization and analysis - optional
try:
    from .circuit_visualization import (
        CircuitDiagram,
        draw_circuit,
        circuit_to_latex,
        export_circuit_image,
    )
except ImportError:
    pass

try:
    from .performance_analysis import (
        compare_classical_quantum,
        benchmark_algorithm,
        plot_complexity_comparison,
        generate_performance_report,
    )
except ImportError:
    pass

__version__ = "1.0.0"
__all__ = [
    # Gates
    "ControlledGate",
    "controlled_z",
    "controlled_u",
    "toffoli",
    "swap_gate",
    "multi_controlled_gate",
    "controlled_phase",
    # Oracles
    "Oracle",
    "deutsch_jozsa_oracle",
    "grover_oracle",
    "create_balanced_function",
    "create_constant_function",
    # Deutsch-Jozsa
    "deutsch_jozsa_algorithm",
    "create_dj_circuit",
    "verify_function_type",
    # Grover
    "grover_search",
    "grover_diffusion",
    "optimal_grover_iterations",
    "amplitude_amplification_step",
    # QFT
    "quantum_fourier_transform",
    "inverse_qft",
    "qft_circuit",
    "controlled_phase_gate",
    # Visualization
    "CircuitDiagram",
    "draw_circuit",
    "circuit_to_latex",
    "export_circuit_image",
    # Performance
    "compare_classical_quantum",
    "benchmark_algorithm",
    "plot_complexity_comparison",
    "generate_performance_report",
]
