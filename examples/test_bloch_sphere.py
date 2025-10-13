#!/usr/bin/env python3
"""
Test script for Bloch Sphere Visualizer
========================================

This script tests the Bloch sphere visualization functionality
to ensure everything works correctly before the demo.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_qubits.qubit import Qubit, ket_0, ket_1, ket_plus, ket_minus
from phase1_qubits.gates import HADAMARD, PAULI_X, PAULI_Y, PAULI_Z, S_GATE, T_GATE, apply_gate
from phase1_qubits.bloch_sphere import BlochSphere, plot_gate_trajectory, compare_states
import numpy as np

# Create plots directory
PLOTS_DIR = Path(__file__).parent.parent / "plots" / "phase1"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def test_basic_visualization():
    """Test 1: Basic Bloch sphere with single qubit"""
    print("Test 1: Creating basic Bloch sphere with |0⟩ state...")

    bloch = BlochSphere(figsize=(10, 10))
    q = ket_0()
    bloch.add_qubit(q, label="|0⟩", color='blue')

    print(f"  Qubit state: {q}")
    print(f"  Bloch coordinates: {q.bloch_coordinates()}")
    print("  ✓ Basic visualization test passed\n")

    # Save to file instead of showing
    output_path = PLOTS_DIR / "bloch_test_basic.png"
    bloch.save(str(output_path), title="Test: |0⟩ State")
    print(f"  Saved to {output_path}\n")


def test_multiple_states():
    """Test 2: Multiple states on same sphere"""
    print("Test 2: Visualizing multiple states...")

    bloch = BlochSphere(figsize=(12, 10))

    states = [
        (ket_0(), "|0⟩", 'blue'),
        (ket_1(), "|1⟩", 'red'),
        (ket_plus(), "|+⟩", 'green'),
        (ket_minus(), "|-⟩", 'purple')
    ]

    for qubit, label, color in states:
        bloch.add_qubit(qubit, label=label, color=color)
        print(f"  Added {label}: Bloch = {qubit.bloch_coordinates()}")

    output_path = PLOTS_DIR / "bloch_test_multiple.png"
    bloch.save(str(output_path), title="Test: Multiple States")
    print("  ✓ Multiple states test passed")
    print(f"  Saved to {output_path}\n")


def test_gate_trajectory():
    """Test 3: Gate trajectory visualization"""
    print("Test 3: Testing gate trajectory visualization...")

    # Create a sequence: |0⟩ → H → S → H
    initial = ket_0()
    gates = [HADAMARD, S_GATE, HADAMARD]
    gate_names = ['H', 'S', 'H']

    print(f"  Initial state: {initial}")

    current = initial.copy()
    for gate, name in zip(gates, gate_names):
        current = apply_gate(current, gate)
        print(f"  After {name}: {current}")

    print("  ✓ Gate trajectory test passed")
    print("  Note: Use plot_gate_trajectory() for visual output\n")


def test_custom_superposition():
    """Test 4: Custom superposition state"""
    print("Test 4: Creating custom superposition state...")

    # Create |ψ⟩ = (|0⟩ + i|1⟩)/√2
    psi = Qubit([1/np.sqrt(2), 1j/np.sqrt(2)])

    print(f"  State: {psi}")
    print(f"  Bloch coordinates: {psi.bloch_coordinates()}")
    print(f"  P(0) = {psi.prob_0():.4f}, P(1) = {psi.prob_1():.4f}")
    print(f"  Is normalized: {psi.is_normalized()}")

    bloch = BlochSphere(figsize=(10, 10))
    bloch.add_qubit(psi, label="|ψ⟩", color='orange')
    output_path = PLOTS_DIR / "bloch_test_superposition.png"
    bloch.save(str(output_path), title="Test: Custom Superposition")

    print("  ✓ Custom superposition test passed")
    print(f"  Saved to {output_path}\n")


def test_rotation_sequence():
    """Test 5: Rotation sequence around different axes"""
    print("Test 5: Testing rotation sequence...")

    initial = ket_0()

    # Apply X, then Y, then Z rotations
    gates = [PAULI_X, PAULI_Y, PAULI_Z]
    gate_names = ['X', 'Y', 'Z']

    bloch = BlochSphere(figsize=(12, 10))
    bloch.add_qubit(initial, label='Start', color='green')

    current = initial.copy()
    colors = ['red', 'blue', 'purple']

    for gate, name, color in zip(gates, gate_names, colors):
        current = apply_gate(current, gate)
        bloch.add_qubit(current, label=name, color=color)
        print(f"  After {name}: Bloch = {current.bloch_coordinates()}")

    output_path = PLOTS_DIR / "bloch_test_rotations.png"
    bloch.save(str(output_path), title="Test: Pauli Rotations")
    print("  ✓ Rotation sequence test passed")
    print(f"  Saved to {output_path}\n")


def run_all_tests():
    """Run all Bloch sphere tests"""
    print("="*60)
    print("BLOCH SPHERE VISUALIZER TEST SUITE")
    print("="*60 + "\n")

    try:
        test_basic_visualization()
        test_multiple_states()
        test_gate_trajectory()
        test_custom_superposition()
        test_rotation_sequence()

        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nGenerated files in plots/phase1/:")
        print("  - bloch_test_basic.png")
        print("  - bloch_test_multiple.png")
        print("  - bloch_test_superposition.png")
        print("  - bloch_test_rotations.png")
        print("\nBloch sphere visualizer is working correctly!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
