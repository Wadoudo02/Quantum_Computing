#!/usr/bin/env python3
"""
Phase 1 Complete Demo
=====================

This script demonstrates ALL Phase 1 functionality:
- Single qubit creation and manipulation
- Quantum gates and gate sequences
- Bloch sphere visualization
- Measurement simulation
- Two-qubit systems and entanglement

Perfect for demonstrating to recruiters at Quantinuum/Riverlane!
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_qubits.qubit import (
    Qubit, ket_0, ket_1, ket_plus, ket_minus, random_qubit
)
from phase1_qubits.gates import (
    HADAMARD, PAULI_X, PAULI_Y, PAULI_Z, S_GATE, T_GATE,
    apply_gate, rx, ry, rz
)
from phase1_qubits.bloch_sphere import (
    BlochSphere, plot_gate_trajectory, compare_states
)
from phase1_qubits.multi_qubit import (
    TwoQubitSystem, bell_phi_plus, bell_phi_minus,
    bell_psi_plus, bell_psi_minus, tensor_product
)
from phase1_qubits.two_qubit_gates import (
    CNOT, SWAP, CZ, apply_gate_to_system
)

# Create plots directory
PLOTS_DIR = Path(__file__).parent.parent / "plots" / "phase1"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def demo_1_basic_qubits():
    """Demo 1: Creating and inspecting qubits"""
    print("="*70)
    print("DEMO 1: Basic Qubit Creation and Properties")
    print("="*70 + "\n")

    # Create basis states
    print("1. Computational basis states:")
    q0 = ket_0()
    q1 = ket_1()
    print(f"   |0⟩ = {q0.state}  →  {q0}")
    print(f"   |1⟩ = {q1.state}  →  {q1}")
    print(f"   Bloch coords: |0⟩ = {q0.bloch_coordinates()}, |1⟩ = {q1.bloch_coordinates()}\n")

    # Create superposition states
    print("2. Hadamard basis states:")
    q_plus = ket_plus()
    q_minus = ket_minus()
    print(f"   |+⟩ = {q_plus}")
    print(f"   |−⟩ = {q_minus}")
    print(f"   P(0) for |+⟩: {q_plus.prob_0():.4f}, P(1): {q_plus.prob_1():.4f}\n")

    # Create custom state
    print("3. Custom superposition:")
    psi = Qubit([np.sqrt(0.7), np.sqrt(0.3)])
    print(f"   |ψ⟩ = {psi}")
    print(f"   P(0) = {psi.prob_0():.4f}, P(1) = {psi.prob_1():.4f}")
    print(f"   Bloch coordinates: {psi.bloch_coordinates()}\n")

    # Visualize on Bloch sphere
    print("4. Visualizing on Bloch sphere...")
    bloch = BlochSphere(figsize=(12, 10))
    bloch.add_qubit(q0, label="|0⟩", color='blue')
    bloch.add_qubit(q1, label="|1⟩", color='red')
    bloch.add_qubit(q_plus, label="|+⟩", color='green')
    bloch.add_qubit(q_minus, label="|−⟩", color='purple')
    bloch.add_qubit(psi, label="|ψ⟩", color='orange')
    output_path = PLOTS_DIR / "demo1_basic_states.png"
    bloch.save(str(output_path), title="Demo 1: Basic Quantum States")
    print(f"   ✓ Saved to {output_path}\n")


def demo_2_quantum_gates():
    """Demo 2: Applying quantum gates"""
    print("="*70)
    print("DEMO 2: Quantum Gate Operations")
    print("="*70 + "\n")

    # Pauli gates
    print("1. Pauli Gates on |0⟩:")
    q = ket_0()
    print(f"   Initial: {q}")
    print(f"   After X: {apply_gate(q, PAULI_X)}")
    print(f"   After Y: {apply_gate(q, PAULI_Y)}")
    print(f"   After Z: {apply_gate(q, PAULI_Z)}")
    print()

    # Hadamard gate
    print("2. Hadamard Gate (creates superposition):")
    q = ket_0()
    print(f"   |0⟩ → H → {apply_gate(q, HADAMARD)}")
    q = ket_1()
    print(f"   |1⟩ → H → {apply_gate(q, HADAMARD)}")
    print()

    # Phase gates
    print("3. Phase Gates:")
    q = ket_plus()
    print(f"   |+⟩ = {q}")
    print(f"   After S: {apply_gate(q, S_GATE)}")
    print(f"   After T: {apply_gate(q, T_GATE)}")
    print()

    # Gate sequence
    print("4. Gate Sequence (|0⟩ → H → S → H):")
    q = ket_0()
    print(f"   Start: {q}")
    q = apply_gate(q, HADAMARD)
    print(f"   After H: {q}")
    q = apply_gate(q, S_GATE)
    print(f"   After S: {q}")
    q = apply_gate(q, HADAMARD)
    print(f"   After H: {q}")
    print()

    # Visualize trajectory
    print("5. Visualizing gate trajectory on Bloch sphere...")
    initial = ket_0()
    gates = [HADAMARD, S_GATE, HADAMARD]
    gate_names = ['H', 'S', 'H']

    bloch = BlochSphere(figsize=(12, 10))
    bloch.add_qubit(initial, label='Start', color='green')

    current = initial.copy()
    colors = ['blue', 'purple', 'red']
    for gate, name, color in zip(gates, gate_names, colors):
        current = apply_gate(current, gate)
        bloch.add_qubit(current, label=name, color=color)

    output_path = PLOTS_DIR / "demo2_gate_trajectory.png"
    bloch.save(str(output_path), title="Demo 2: Gate Trajectory")
    print(f"   ✓ Saved to {output_path}\n")


def demo_3_measurements():
    """Demo 3: Quantum measurements"""
    print("="*70)
    print("DEMO 3: Quantum Measurements")
    print("="*70 + "\n")

    # Measure deterministic states
    print("1. Measuring deterministic states:")
    q0 = ket_0()
    results = q0.measure(shots=10)
    print(f"   |0⟩ measured 10 times: {results}")

    q1 = ket_1()
    results = q1.measure(shots=10)
    print(f"   |1⟩ measured 10 times: {results}")
    print()

    # Measure superposition
    print("2. Measuring superposition state:")
    q_plus = ket_plus()
    print(f"   State: {q_plus}")
    print(f"   Theoretical: P(0) = {q_plus.prob_0():.4f}, P(1) = {q_plus.prob_1():.4f}")

    results = q_plus.measure(shots=1000)
    zeros = np.sum(results == 0)
    ones = np.sum(results == 1)
    print(f"   1000 measurements: {zeros} zeros, {ones} ones")
    print(f"   Experimental: P(0) = {zeros/1000:.4f}, P(1) = {ones/1000:.4f}")
    print()

    # Measure custom state
    print("3. Measuring custom state (70-30 split):")
    psi = Qubit([np.sqrt(0.7), np.sqrt(0.3)])
    print(f"   State: {psi}")
    print(f"   Theoretical: P(0) = {psi.prob_0():.4f}, P(1) = {psi.prob_1():.4f}")

    results = psi.measure(shots=1000)
    zeros = np.sum(results == 0)
    ones = np.sum(results == 1)
    print(f"   1000 measurements: {zeros} zeros, {ones} ones")
    print(f"   Experimental: P(0) = {zeros/1000:.4f}, P(1) = {ones/1000:.4f}")
    print()


def demo_4_rotation_gates():
    """Demo 4: Rotation gates"""
    print("="*70)
    print("DEMO 4: Rotation Gates")
    print("="*70 + "\n")

    print("1. Rotation gates around X, Y, Z axes:")
    q = ket_0()

    # Rotate around X axis (π/2)
    angle = np.pi / 2
    q_rx = apply_gate(q, rx(angle))
    print(f"   R_x(π/2)|0⟩ = {q_rx}")
    print(f"   Bloch coords: {q_rx.bloch_coordinates()}")

    # Rotate around Y axis (π/2)
    q_ry = apply_gate(q, ry(angle))
    print(f"   R_y(π/2)|0⟩ = {q_ry}")
    print(f"   Bloch coords: {q_ry.bloch_coordinates()}")

    # Rotate around Z axis (π/2)
    q_rz = apply_gate(q, rz(angle))
    print(f"   R_z(π/2)|0⟩ = {q_rz}")
    print(f"   Bloch coords: {q_rz.bloch_coordinates()}")
    print()

    # Visualize rotations
    print("2. Visualizing rotations on Bloch sphere...")
    bloch = BlochSphere(figsize=(12, 10))
    bloch.add_qubit(q, label='|0⟩', color='gray')
    bloch.add_qubit(q_rx, label='Rx(π/2)', color='red')
    bloch.add_qubit(q_ry, label='Ry(π/2)', color='green')
    bloch.add_qubit(q_rz, label='Rz(π/2)', color='blue')
    output_path = PLOTS_DIR / "demo4_rotations.png"
    bloch.save(str(output_path), title="Demo 4: Rotation Gates")
    print(f"   ✓ Saved to {output_path}\n")


def demo_5_two_qubit_systems():
    """Demo 5: Two-qubit systems and entanglement"""
    print("="*70)
    print("DEMO 5: Two-Qubit Systems and Entanglement")
    print("="*70 + "\n")

    # Create product state
    print("1. Product state (separable):")
    q0 = ket_0()
    q1 = ket_0()
    system = tensor_product(q0, q1)
    print(f"   |00⟩ = |0⟩ ⊗ |0⟩")
    print(f"   State vector: {system.state}")
    print(f"   Is entangled? {system.is_entangled()}")
    print()

    # Create Bell state
    print("2. Bell state |Φ+⟩ (maximally entangled):")
    bell = bell_phi_plus()
    print(f"   |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print(f"   State vector: {bell.state}")
    print(f"   Is entangled? {bell.is_entangled()}")
    print(f"   Entanglement entropy: {bell.entanglement_entropy():.4f}")
    print()

    # CNOT gate creates entanglement
    print("3. Creating entanglement with CNOT:")
    q0 = ket_0()
    q1 = ket_0()
    system = tensor_product(q0, q1)
    print(f"   Initial: |00⟩ (separable)")
    print(f"   Is entangled? {system.is_entangled()}")

    # Apply Hadamard to first qubit
    from phase1_qubits.two_qubit_gates import apply_single_qubit_gate
    system = apply_single_qubit_gate(system, HADAMARD, qubit_index=0)
    print(f"   After H⊗I: (|0⟩+|1⟩)/√2 ⊗ |0⟩")
    print(f"   Is entangled? {system.is_entangled()}")

    # Apply CNOT
    system = apply_gate_to_system(system, CNOT)
    print(f"   After CNOT: {system.state}")
    print(f"   Is entangled? {system.is_entangled()}")
    print(f"   Created Bell state |Φ+⟩!\n")

    # All Bell states
    print("4. All four Bell states:")
    bell_states = [
        (bell_phi_plus(), "|Φ+⟩", "(|00⟩ + |11⟩)/√2"),
        (bell_phi_minus(), "|Φ−⟩", "(|00⟩ − |11⟩)/√2"),
        (bell_psi_plus(), "|Ψ+⟩", "(|01⟩ + |10⟩)/√2"),
        (bell_psi_minus(), "|Ψ−⟩", "(|01⟩ − |10⟩)/√2"),
    ]

    for state, name, formula in bell_states:
        entropy = state.entanglement_entropy()
        print(f"   {name} = {formula}")
        print(f"   State: {state.state}")
        print(f"   Entanglement: {entropy:.4f}")
        print()


def demo_6_two_qubit_gates():
    """Demo 6: Two-qubit gates"""
    print("="*70)
    print("DEMO 6: Two-Qubit Gate Operations")
    print("="*70 + "\n")

    # CNOT gate
    print("1. CNOT Gate (Controlled-NOT):")
    basis_states = [
        (ket_0(), ket_0(), "|00⟩"),
        (ket_0(), ket_1(), "|01⟩"),
        (ket_1(), ket_0(), "|10⟩"),
        (ket_1(), ket_1(), "|11⟩"),
    ]

    for q0, q1, label in basis_states:
        system = tensor_product(q0, q1)
        after = apply_gate_to_system(system, CNOT)
        print(f"   CNOT {label} → {after.state}")

    print()

    # SWAP gate
    print("2. SWAP Gate:")
    system = tensor_product(ket_0(), ket_1())  # |01⟩
    print(f"   Before SWAP: |01⟩ = {system.state}")
    after = apply_gate_to_system(system, SWAP)
    print(f"   After SWAP:  |10⟩ = {after.state}")
    print()

    # Controlled-Z gate
    print("3. Controlled-Z Gate:")
    system = tensor_product(ket_plus(), ket_plus())  # |++⟩
    print(f"   CZ on |++⟩")
    after = apply_gate_to_system(system, CZ)
    print(f"   Result: {after.state}")
    print()


def run_all_demos():
    """Run all Phase 1 demonstrations"""
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE DEMONSTRATION")
    print("Quantum Computing Implementation - Quantinuum/Riverlane Recruitment")
    print("="*70 + "\n")

    demo_1_basic_qubits()
    input("Press Enter to continue to Demo 2...")

    demo_2_quantum_gates()
    input("Press Enter to continue to Demo 3...")

    demo_3_measurements()
    input("Press Enter to continue to Demo 4...")

    demo_4_rotation_gates()
    input("Press Enter to continue to Demo 5...")

    demo_5_two_qubit_systems()
    input("Press Enter to continue to Demo 6...")

    demo_6_two_qubit_gates()

    print("="*70)
    print("ALL DEMOS COMPLETED!")
    print("="*70)
    print("\nGenerated visualizations in plots/phase1/:")
    print("  - demo1_basic_states.png")
    print("  - demo2_gate_trajectory.png")
    print("  - demo4_rotations.png")
    print("\nNext steps:")
    print("  - Run the Streamlit app: cd src/phase1_qubits && streamlit run app.py")
    print("  - Review Phase 1 completion in quantum_master_plan.md")
    print("  - Move to Phase 2: Entanglement and Bell States")
    print()


if __name__ == "__main__":
    run_all_demos()
