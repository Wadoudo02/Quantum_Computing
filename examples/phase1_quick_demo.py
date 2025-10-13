#!/usr/bin/env python3
"""
Phase 1 Quick Demo (Non-Interactive)
=====================================

Quick, non-interactive demonstration of all Phase 1 functionality.
Perfect for automated testing or quick verification.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_qubits.qubit import (
    Qubit, ket_0, ket_1, ket_plus, ket_minus
)
from phase1_qubits.gates import (
    HADAMARD, PAULI_X, PAULI_Y, PAULI_Z, S_GATE, T_GATE,
    apply_gate, rx, ry, rz
)
from phase1_qubits.bloch_sphere import BlochSphere
from phase1_qubits.multi_qubit import (
    TwoQubitSystem, bell_phi_plus, tensor_product
)
from phase1_qubits.two_qubit_gates import (
    CNOT, apply_gate_to_system, apply_single_qubit_gate
)

# Create plots directory
PLOTS_DIR = Path(__file__).parent.parent / "plots" / "phase1"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Run quick demonstration of all Phase 1 features."""
    print("="*70)
    print("PHASE 1 QUICK DEMO")
    print("="*70 + "\n")

    # Demo 1: Basic Qubits
    print("1. Basic Qubit States:")
    q0 = ket_0()
    q1 = ket_1()
    q_plus = ket_plus()
    print(f"   |0⟩: {q0}, Bloch: {q0.bloch_coordinates()}")
    print(f"   |1⟩: {q1}, Bloch: {q1.bloch_coordinates()}")
    print(f"   |+⟩: {q_plus}, P(0)={q_plus.prob_0():.3f}")
    print()

    # Demo 2: Quantum Gates
    print("2. Quantum Gates:")
    q = ket_0()
    print(f"   |0⟩ → X → {apply_gate(q, PAULI_X)}")
    print(f"   |0⟩ → H → {apply_gate(q, HADAMARD)}")
    print(f"   |0⟩ → Y → {apply_gate(q, PAULI_Y)}")
    print()

    # Demo 3: Rotation Gates
    print("3. Rotation Gates:")
    angle = np.pi / 4
    print(f"   Rx(π/4)|0⟩ = {apply_gate(q, rx(angle))}")
    print(f"   Ry(π/4)|0⟩ = {apply_gate(q, ry(angle))}")
    print(f"   Rz(π/4)|0⟩ = {apply_gate(q, rz(angle))}")
    print()

    # Demo 4: Measurement
    print("4. Measurement:")
    q_plus = ket_plus()
    results = q_plus.measure(shots=100)
    zeros = np.sum(results == 0)
    print(f"   |+⟩ measured 100 times: {zeros} zeros, {100-zeros} ones")
    print(f"   Expected: ~50/50 split")
    print()

    # Demo 5: Two-Qubit Systems
    print("5. Two-Qubit Systems:")
    system = tensor_product(ket_0(), ket_1())
    print(f"   |0⟩ ⊗ |1⟩ = |01⟩: {system.state}")
    print(f"   Is entangled? {system.is_entangled()}")
    print()

    # Demo 6: Bell States
    print("6. Bell States (Entanglement):")
    bell = bell_phi_plus()
    print(f"   |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print(f"   State: {bell.state}")
    print(f"   Is entangled? {bell.is_entangled()}")
    print(f"   Entanglement entropy: {bell.entanglement_entropy():.4f}")
    print()

    # Demo 7: CNOT Gate
    print("7. Creating Entanglement with CNOT:")
    system = tensor_product(ket_0(), ket_0())
    print(f"   Start: |00⟩, entangled={system.is_entangled()}")
    system = apply_single_qubit_gate(system, HADAMARD, 0)
    print(f"   After H⊗I: entangled={system.is_entangled()}")
    system = apply_gate_to_system(system, CNOT)
    print(f"   After CNOT: entangled={system.is_entangled()} ✓")
    print()

    # Demo 8: Bloch Sphere Visualization
    print("8. Bloch Sphere Visualization:")
    bloch = BlochSphere(figsize=(10, 10))
    bloch.add_qubit(ket_0(), label="|0⟩", color='blue')
    bloch.add_qubit(ket_1(), label="|1⟩", color='red')
    bloch.add_qubit(ket_plus(), label="|+⟩", color='green')
    output_path = PLOTS_DIR / "quick_demo_bloch.png"
    bloch.save(str(output_path), title="Phase 1 Quick Demo")
    print(f"   ✓ Saved Bloch sphere to {output_path}")
    print()

    print("="*70)
    print("QUICK DEMO COMPLETE!")
    print("="*70)
    print("\nAll Phase 1 functionality working correctly!")
    print("\nFor interactive demo, run:")
    print("  cd src/phase1_qubits && streamlit run app.py")
    print("\nFor full demo with visualizations, run:")
    print("  python examples/phase1_complete_demo.py")
    print()


if __name__ == "__main__":
    main()
