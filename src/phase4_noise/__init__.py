# -*- coding: utf-8 -*-
"""
Phase 4: Quantum Noise and Decoherence

This module implements density matrix formalism, quantum noise channels,
and decoherence simulation to understand why quantum computers are so challenging to build.

Based on Imperial College Quantum Information Theory Notes - Section 4.1-4.2

Key Components:
- Density matrices for mixed states
- Kraus operator representation of noise
- Six fundamental quantum channels (bit-flip, phase-flip, depolarizing, amplitude/phase damping)
- T1/T2 decoherence simulation
- Impact analysis on quantum algorithms and Bell states

Author: Wadoud Charbak
For: Quantinuum & Riverlane recruitment
"""

# Core density matrix operations
from .density_matrix import (
    DensityMatrix,
    pure_state_density_matrix,
    mixed_state_density_matrix,
    purity,
    is_pure,
    fidelity,
    trace_distance,
    bloch_vector,
)

# Quantum noise channels
from .quantum_channels import (
    apply_channel,
    bit_flip_channel,
    phase_flip_channel,
    bit_phase_flip_channel,
    depolarizing_channel,
    amplitude_damping_channel,
    phase_damping_channel,
    verify_kraus_completeness,
)

# Decoherence simulation
from .decoherence import (
    DecoherenceSimulator,
    simulate_t1_decay,
    simulate_t2_decay,
    simulate_combined_decay,
    ramsey_experiment,
)

__all__ = [
    # Density matrix
    "DensityMatrix",
    "pure_state_density_matrix",
    "mixed_state_density_matrix",
    "purity",
    "is_pure",
    "fidelity",
    "trace_distance",
    "bloch_vector",
    # Noise channels
    "apply_channel",
    "bit_flip_channel",
    "phase_flip_channel",
    "bit_phase_flip_channel",
    "depolarizing_channel",
    "amplitude_damping_channel",
    "phase_damping_channel",
    "verify_kraus_completeness",
    # Decoherence
    "DecoherenceSimulator",
    "simulate_t1_decay",
    "simulate_t2_decay",
    "simulate_combined_decay",
    "ramsey_experiment",
]

__version__ = "1.0.0"
