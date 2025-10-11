"""
Architecture Demo
=================

Shows how the Qubit, TwoQubitSystem, and gate classes work together.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_qubits.qubit import ket_0, ket_1, ket_plus
from phase1_qubits.gates import hadamard, x_gate
from phase1_qubits.multi_qubit import TwoQubitSystem, bell_phi_plus
from phase1_qubits.two_qubit_gates import CNOT, SWAP, apply_gate_to_system


def demo_single_qubit():
    """Demonstrate single qubit operations."""
    print("\n" + "="*60)
    print("SINGLE QUBIT OPERATIONS")
    print("="*60)

    q = ket_0()
    print(f"\nStarting state: {q}")
    print(f"Bloch coordinates: {q.bloch_coordinates()}")

    q = hadamard(q)
    print(f"\nAfter Hadamard: {q}")
    print(f"Bloch coordinates: {q.bloch_coordinates()}")

    print(f"\nMeasurement probabilities:")
    print(f"  P(0) = {q.prob_0():.3f}")
    print(f"  P(1) = {q.prob_1():.3f}")

    results = q.measure(shots=100)
    print(f"\nMeasured 100 times:")
    print(f"  Got 0: {sum(results == 0)} times")
    print(f"  Got 1: {sum(results == 1)} times")


def demo_two_qubit_system():
    """Demonstrate two-qubit system."""
    print("\n" + "="*60)
    print("TWO-QUBIT SYSTEM")
    print("="*60)

    # Create from single qubits
    q1 = ket_0()
    q2 = ket_1()

    sys = TwoQubitSystem.from_single_qubits(q1, q2)
    print(f"\n|0⟩ ⊗ |1⟩ = {sys}")
    print(f"Is entangled? {sys.is_entangled()}")

    # Create Bell state
    bell = bell_phi_plus()
    print(f"\nBell state |Φ+⟩ = {bell}")
    print(f"Is entangled? {bell.is_entangled()}")
    print(f"Entanglement entropy: {bell.entanglement_entropy():.3f}")

    # Schmidt decomposition
    coeffs, _, _ = bell.schmidt_decomposition()
    print(f"Schmidt coefficients: {coeffs}")
    print(f"Schmidt rank: {len(coeffs)}")

    # Reduced density matrix
    rho = bell.reduced_density_matrix(0)
    print(f"\nReduced density matrix for qubit 1:")
    print(rho)
    print("^ This is maximally mixed! (0.5 on diagonal)")


def demo_creating_entanglement():
    """Show how CNOT creates entanglement."""
    print("\n" + "="*60)
    print("CREATING ENTANGLEMENT WITH CNOT")
    print("="*60)

    # Start with product state
    q1 = hadamard(ket_0())  # |+⟩
    q2 = ket_0()

    sys = TwoQubitSystem.from_single_qubits(q1, q2)
    print(f"\nInitial: |+⟩ ⊗ |0⟩ = {sys}")
    print(f"Is entangled? {sys.is_entangled()}")

    # Apply CNOT
    sys = apply_gate_to_system(sys, CNOT)
    print(f"\nAfter CNOT: {sys}")
    print(f"Is entangled? {sys.is_entangled()}")
    print(f"Entanglement entropy: {sys.entanglement_entropy():.3f}")

    print("\n✨ We created a Bell state!")


def demo_swap_gate():
    """Demonstrate SWAP gate."""
    print("\n" + "="*60)
    print("SWAP GATE")
    print("="*60)

    # Create |01⟩
    sys = TwoQubitSystem.from_single_qubits(ket_0(), ket_1())
    print(f"\nInitial state: {sys}")

    # Apply SWAP
    sys = apply_gate_to_system(sys, SWAP)
    print(f"After SWAP: {sys}")
    print("^ Qubits exchanged!")


def demo_measurement():
    """Demonstrate measurement on entangled states."""
    print("\n" + "="*60)
    print("MEASUREMENT ON ENTANGLED STATES")
    print("="*60)

    bell = bell_phi_plus()
    print(f"\nState: {bell}")

    # Measure both qubits 10 times
    results = bell.measure(shots=10)
    print("\n10 measurements (both qubits):")
    for i, (q1, q2) in enumerate(results):
        print(f"  Shot {i+1}: qubit1={q1}, qubit2={q2}")

    print("\nNotice: When qubit1=0, qubit2=0")
    print("        When qubit1=1, qubit2=1")
    print("They're always correlated! (That's entanglement)")

    # Partial measurement
    print("\n\nMeasuring ONLY first qubit:")
    results = bell.measure_qubit(0, shots=10)
    print(f"Results: {results}")
    print("Should be roughly 50% 0s and 50% 1s")


def demo_quantum_circuit():
    """Build a simple quantum circuit."""
    print("\n" + "="*60)
    print("BUILDING A QUANTUM CIRCUIT")
    print("="*60)

    print("\nCircuit:")
    print("  |0⟩ ─H─●─SWAP─")
    print("         │  │")
    print("  |0⟩ ───X─SWAP─")
    print()

    # Step 1: Hadamard on first qubit
    q1 = hadamard(ket_0())
    q2 = ket_0()
    sys = TwoQubitSystem.from_single_qubits(q1, q2)
    print(f"After H:    {sys}")

    # Step 2: CNOT
    sys = apply_gate_to_system(sys, CNOT)
    print(f"After CNOT: {sys}")

    # Step 3: SWAP
    sys = apply_gate_to_system(sys, SWAP)
    print(f"After SWAP: {sys}")

    print(f"\nFinal state entangled? {sys.is_entangled()}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUANTUM ARCHITECTURE DEMONSTRATION")
    print("="*60)

    demo_single_qubit()
    demo_two_qubit_system()
    demo_creating_entanglement()
    demo_swap_gate()
    demo_measurement()
    demo_quantum_circuit()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Architecture:
  • Qubit class → Single qubits
  • TwoQubitSystem class → Two qubits + entanglement
  • gates.py → Single-qubit gates (2×2 matrices)
  • two_qubit_gates.py → Two-qubit gates (4×4 matrices)

Key concepts demonstrated:
  1. Creating superposition (Hadamard)
  2. Creating entanglement (CNOT)
  3. Measuring entanglement (Schmidt decomposition)
  4. Quantum correlations (Bell states)
  5. Building circuits (gate sequences)

Next steps:
  • Bloch sphere visualization
  • Streamlit interactive app
  • Phase 2: More entanglement properties
""")
