"""
Phase 5: Quantum Error Correction

This package implements quantum error correction codes and provides tools for
analyzing error correction performance.

Modules:
    bit_flip_code: 3-qubit bit-flip error correction code
    shor_code: Shor's 9-qubit universal error correction code
    stabilizers: Stabilizer formalism framework
    error_analysis: Error threshold and overhead analysis
    visualizations: Plotting tools for error correction

Author: Quantum Computing Learning Project
Phase: 5 - Error Correction
"""

from .bit_flip_code import BitFlipCode, demonstrate_bit_flip_code
from .shor_code import ShorCode, demonstrate_shor_code
from .stabilizers import (
    PauliOperator,
    StabilizerCode,
    BitFlipStabilizerCode,
    ShorStabilizerCode,
    FiveQubitCode,
    demonstrate_stabilizers
)
from .error_analysis import (
    ErrorAnalyzer,
    ThresholdCalculator,
    ErrorCorrectionPerformance,
    demonstrate_error_analysis
)
from .visualizations import ECCVisualizer, demo_visualizations

__all__ = [
    # 3-Qubit Bit-Flip Code
    'BitFlipCode',
    'demonstrate_bit_flip_code',

    # Shor's 9-Qubit Code
    'ShorCode',
    'demonstrate_shor_code',

    # Stabilizer Formalism
    'PauliOperator',
    'StabilizerCode',
    'BitFlipStabilizerCode',
    'ShorStabilizerCode',
    'FiveQubitCode',
    'demonstrate_stabilizers',

    # Error Analysis
    'ErrorAnalyzer',
    'ThresholdCalculator',
    'ErrorCorrectionPerformance',
    'demonstrate_error_analysis',

    # Visualizations
    'ECCVisualizer',
    'demo_visualizations',
]

__version__ = '1.0.0'
__author__ = 'Quantum Computing Learning Project'
__phase__ = 5
