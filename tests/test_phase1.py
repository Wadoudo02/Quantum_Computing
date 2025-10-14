"""Quick tests to verify Phase 1 is working"""
import sys
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from phase1_qubits.qubit import Qubit, ket_0, ket_1, ket_plus
from phase1_qubits.gates import *
import numpy as np

def test_qubit_creation():
    """Test basic qubit creation"""
    q = Qubit([1, 0])
    assert q.prob_0() == 1.0
    assert q.prob_1() == 0.0
    print("✅ Qubit creation works!")

def test_pauli_x():
    """Test X gate flips |0⟩ to |1⟩"""
    q = ket_0()
    q_flipped = x_gate(q)
    assert q_flipped.prob_1() > 0.99
    print("✅ Pauli X gate works!")

def test_hadamard():
    """Test Hadamard creates superposition"""
    q = ket_0()
    q_super = hadamard(q)
    # Should be 50/50 superposition
    assert abs(q_super.prob_0() - 0.5) < 0.01
    assert abs(q_super.prob_1() - 0.5) < 0.01
    print("✅ Hadamard gate works!")

def test_bloch_sphere():
    """Test Bloch sphere coordinates"""
    q = ket_0()
    x, y, z = q.bloch_coordinates()
    # |0⟩ should be at north pole (0, 0, 1)
    assert abs(z - 1.0) < 0.01
    print("✅ Bloch sphere coordinates work!")

if __name__ == "__main__":
    test_qubit_creation()
    test_pauli_x()
    test_hadamard()
    test_bloch_sphere()
    print("\n🎉 All basic tests passed!")