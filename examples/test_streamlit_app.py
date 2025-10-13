#!/usr/bin/env python3
"""
Test script for Streamlit App Components
=========================================

This script tests the core functionality of the Streamlit app
to ensure all components work correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_qubits.qubit import Qubit, ket_0, ket_1, ket_plus, ket_minus
from phase1_qubits.gates import HADAMARD, PAULI_X, PAULI_Y, PAULI_Z, apply_gate
from phase1_qubits.bloch_sphere import BlochSphere
import numpy as np

# Create plots directory
PLOTS_DIR = Path(__file__).parent.parent / "plots" / "phase1"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def test_qubit_creation_from_angles():
    """Test creating qubits from theta/phi angles (as used in app)"""
    print("Test 1: Creating qubits from Bloch angles...")

    # Test cases: (theta, phi, expected_description)
    test_cases = [
        (0, 0, "|0⟩"),
        (np.pi, 0, "|1⟩"),
        (np.pi/2, 0, "|+⟩"),
        (np.pi/2, np.pi, "|-⟩"),
    ]

    for theta, phi, desc in test_cases:
        # Create qubit from angles (as in Streamlit app)
        alpha = np.cos(theta / 2)
        beta = np.exp(1j * phi) * np.sin(theta / 2)
        qubit = Qubit([alpha, beta], normalize=False)

        print(f"  θ={theta:.2f}, φ={phi:.2f} → {qubit} (expected: {desc})")

        # Verify it's normalized
        assert qubit.is_normalized(), f"Qubit not normalized for θ={theta}, φ={phi}"

        # Verify Bloch coordinates
        x, y, z = qubit.bloch_coordinates()
        print(f"    Bloch: ({x:.3f}, {y:.3f}, {z:.3f})")

    print("  ✓ Angle-based qubit creation test passed\n")


def test_gate_application_sequence():
    """Test applying sequences of gates (as used in app)"""
    print("Test 2: Testing gate application sequences...")

    initial = ket_0()
    gate_sequence = [
        ("H", HADAMARD),
        ("X", PAULI_X),
        ("H", HADAMARD),
    ]

    current = initial.copy()
    print(f"  Initial: {current}")

    for name, gate in gate_sequence:
        current = apply_gate(current, gate)
        print(f"  After {name}: {current}")
        assert current.is_normalized(), f"State not normalized after {name}"

    print("  ✓ Gate sequence test passed\n")


def test_measurement_simulation():
    """Test measurement functionality (as used in app)"""
    print("Test 3: Testing measurement simulation...")

    # Test deterministic case: |0⟩ should always give 0
    q0 = ket_0()
    results = q0.measure(shots=100)
    assert all(r == 0 for r in results), "Measurement of |0⟩ should always give 0"
    print(f"  |0⟩ measured 100 times: {np.sum(results == 0)} zeros, {np.sum(results == 1)} ones ✓")

    # Test deterministic case: |1⟩ should always give 1
    q1 = ket_1()
    results = q1.measure(shots=100)
    assert all(r == 1 for r in results), "Measurement of |1⟩ should always give 1"
    print(f"  |1⟩ measured 100 times: {np.sum(results == 0)} zeros, {np.sum(results == 1)} ones ✓")

    # Test probabilistic case: |+⟩ should give ~50/50
    q_plus = ket_plus()
    results = q_plus.measure(shots=1000)
    zeros = np.sum(results == 0)
    ones = np.sum(results == 1)
    ratio = zeros / 1000
    print(f"  |+⟩ measured 1000 times: {zeros} zeros, {ones} ones (ratio: {ratio:.2f})")
    assert 0.4 < ratio < 0.6, "Expected roughly 50/50 distribution"

    print("  ✓ Measurement simulation test passed\n")


def test_common_states():
    """Test all common states used in app"""
    print("Test 4: Testing common quantum states...")

    states = {
        "|0⟩": ket_0(),
        "|1⟩": ket_1(),
        "|+⟩": ket_plus(),
        "|-⟩": ket_minus(),
    }

    for name, qubit in states.items():
        x, y, z = qubit.bloch_coordinates()
        p0, p1 = qubit.prob_0(), qubit.prob_1()
        print(f"  {name}: Bloch=({x:.3f}, {y:.3f}, {z:.3f}), P(0)={p0:.3f}, P(1)={p1:.3f}")
        assert qubit.is_normalized(), f"{name} not normalized"

    print("  ✓ Common states test passed\n")


def test_bloch_visualization_integration():
    """Test that Bloch sphere works with different qubit states"""
    print("Test 5: Testing Bloch sphere integration...")

    bloch = BlochSphere(figsize=(10, 10))

    # Create a custom state
    theta = np.pi / 3  # 60 degrees
    phi = np.pi / 4    # 45 degrees

    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    qubit = Qubit([alpha, beta])

    bloch.add_qubit(qubit, label="Custom", color='red')

    # Verify coordinates
    x, y, z = qubit.bloch_coordinates()
    print(f"  Custom state (θ={theta:.2f}, φ={phi:.2f}):")
    print(f"    Bloch coordinates: ({x:.3f}, {y:.3f}, {z:.3f})")
    print(f"    State: {qubit}")

    # Save visualization
    output_path = PLOTS_DIR / "bloch_test_app_integration.png"
    bloch.save(str(output_path), title="App Integration Test")
    print(f"  Saved to {output_path}")
    print("  ✓ Bloch sphere integration test passed\n")


def test_gate_laboratory_workflow():
    """Test the complete workflow of the Gate Laboratory mode"""
    print("Test 6: Testing Gate Laboratory workflow...")

    # Simulate user selecting initial state and gate
    initial_state = ket_0()
    selected_gate = "H"
    gate_matrix = HADAMARD

    print(f"  Initial state: {initial_state}")
    print(f"  Applying gate: {selected_gate}")

    # Apply gate
    final_state = apply_gate(initial_state, gate_matrix)
    print(f"  Final state: {final_state}")

    # Create before/after visualization
    bloch_before = BlochSphere(figsize=(8, 8))
    bloch_before.add_qubit(initial_state, label="Before", color='blue')

    bloch_after = BlochSphere(figsize=(8, 8))
    bloch_after.add_qubit(final_state, label="After", color='red')

    print("  ✓ Gate Laboratory workflow test passed\n")


def run_all_tests():
    """Run all Streamlit app component tests"""
    print("="*60)
    print("STREAMLIT APP COMPONENT TEST SUITE")
    print("="*60 + "\n")

    try:
        test_qubit_creation_from_angles()
        test_gate_application_sequence()
        test_measurement_simulation()
        test_common_states()
        test_bloch_visualization_integration()
        test_gate_laboratory_workflow()

        print("="*60)
        print("ALL COMPONENT TESTS PASSED ✓")
        print("="*60)
        print("\nStreamlit app core functionality is working correctly!")
        print("\nTo run the Streamlit app:")
        print("  cd src/phase1_qubits")
        print("  streamlit run app.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
