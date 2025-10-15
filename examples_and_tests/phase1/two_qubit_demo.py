"""
Two-Qubit Gates Demonstration
==============================

Simple examples showing how CNOT, SWAP, and other two-qubit gates work.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from phase1_qubits.two_qubit_gates import (
    CNOT, SWAP, CZ,
    apply_two_qubit_gate,
    tensor_product,
    KET_00, KET_01, KET_10, KET_11
)


def print_state(state: np.ndarray, label: str = ""):
    """Pretty print a two-qubit state."""
    basis = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

    print(f"\n{label}")
    print("State vector: [", end="")
    for i, amp in enumerate(state):
        if i > 0:
            print(", ", end="")
        # Show real part if complex is negligible
        if np.abs(amp.imag) < 1e-10:
            print(f"{amp.real:.3f}", end="")
        else:
            print(f"{amp:.3f}", end="")
    print("]")

    print("In basis notation:")
    for i, amp in enumerate(state):
        if np.abs(amp) > 1e-10:
            if np.abs(amp.imag) < 1e-10:
                print(f"  {amp.real:+.3f} {basis[i]}")
            else:
                print(f"  {amp:.3f} {basis[i]}")


def demo_cnot():
    """Demonstrate CNOT gate on all basis states."""
    print("\n" + "="*60)
    print("CNOT GATE DEMONSTRATION")
    print("="*60)
    print("\nRemember: CNOT flips the second qubit if the first is |1⟩")

    # Test all basis states
    states = [
        (KET_00, "|00⟩"),
        (KET_01, "|01⟩"),
        (KET_10, "|10⟩"),
        (KET_11, "|11⟩")
    ]

    for state, name in states:
        result = apply_two_qubit_gate(state, CNOT)
        print(f"\nCNOT {name} →", end="")
        print_state(result, "")


def demo_swap():
    """Demonstrate SWAP gate."""
    print("\n" + "="*60)
    print("SWAP GATE DEMONSTRATION")
    print("="*60)
    print("\nRemember: SWAP exchanges the two qubits")

    # Test interesting cases
    states = [
        (KET_00, "|00⟩ (nothing to swap)"),
        (KET_01, "|01⟩ (first gets 1, second gets 0)"),
        (KET_10, "|10⟩ (first gets 0, second gets 1)"),
        (KET_11, "|11⟩ (nothing to swap)")
    ]

    for state, name in states:
        result = apply_two_qubit_gate(state, SWAP)
        print(f"\nSWAP {name} →", end="")
        print_state(result, "")


def demo_entanglement():
    """Show how CNOT creates entanglement."""
    print("\n" + "="*60)
    print("ENTANGLEMENT WITH CNOT")
    print("="*60)
    print("\nStarting with |+⟩ ⊗ |0⟩ = (|0⟩+|1⟩)/√2 ⊗ |0⟩")

    # Create |+⟩ = (|0⟩ + |1⟩)/√2
    ket_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ket_0 = np.array([1, 0], dtype=complex)

    # Tensor product
    initial_state = tensor_product(ket_plus, ket_0)
    print_state(initial_state, "\nInitial state |+⟩⊗|0⟩:")

    # Apply CNOT
    entangled_state = apply_two_qubit_gate(initial_state, CNOT)
    print_state(entangled_state, "\nAfter CNOT (Bell state |Φ+⟩):")

    print("\n✨ This is maximally entangled! Can't write as |ψ⟩⊗|φ⟩")
    print("   It's the famous Bell state: (|00⟩ + |11⟩)/√2")


def demo_cz():
    """Demonstrate Controlled-Z gate."""
    print("\n" + "="*60)
    print("CONTROLLED-Z GATE DEMONSTRATION")
    print("="*60)
    print("\nRemember: CZ adds minus sign only to |11⟩")

    states = [
        (KET_00, "|00⟩"),
        (KET_01, "|01⟩"),
        (KET_10, "|10⟩"),
        (KET_11, "|11⟩ (gets minus sign!)")
    ]

    for state, name in states:
        result = apply_two_qubit_gate(state, CZ)
        print(f"\nCZ {name} →", end="")
        print_state(result, "")


def demo_cnot_properties():
    """Show that CNOT is self-inverse."""
    print("\n" + "="*60)
    print("CNOT PROPERTIES")
    print("="*60)
    print("\nCNOT is self-inverse: CNOT · CNOT = Identity")

    initial = KET_10
    print_state(initial, "\nStarting with |10⟩:")

    after_first = apply_two_qubit_gate(initial, CNOT)
    print_state(after_first, "\nAfter first CNOT:")

    after_second = apply_two_qubit_gate(after_first, CNOT)
    print_state(after_second, "\nAfter second CNOT (back to original!):")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TWO-QUBIT QUANTUM GATES DEMO")
    print("="*60)
    print("\nThis demonstrates the essential two-qubit gates from")
    print("Imperial College notes Section 2.2 and 2.4")

    demo_cnot()
    demo_swap()
    demo_cz()
    demo_entanglement()
    demo_cnot_properties()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. CNOT (Controlled-NOT):
   - Flips second qubit if first is |1⟩
   - Creates entanglement
   - Universal for quantum computing

2. SWAP:
   - Exchanges two qubits
   - Simple but useful for routing

3. CZ (Controlled-Z):
   - Adds minus sign to |11⟩
   - Symmetric (doesn't matter which is control)

4. These gates are the building blocks of quantum circuits!
""")
