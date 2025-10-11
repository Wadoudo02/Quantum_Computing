# Quantum Gates Summary

Quick reference for the quantum gates in your implementation.

## Single-Qubit Gates (Phase 1)

Located in: `src/phase1_single_qubits/gates.py`

### Pauli Gates

**NOT Gate (Pauli X)**
- **What it does**: Flips qubit: |0⟩ ↔ |1⟩
- **Think of it as**: Classical NOT gate
- **Matrix**: [[0, 1], [1, 0]]
- **Use**: `x_gate(qubit)` or `apply_gate(qubit, PAULI_X)`

**Pauli Y**
- **What it does**: Bit flip + phase flip
- **Matrix**: [[0, -i], [i, 0]]
- **Use**: `y_gate(qubit)`

**Pauli Z**
- **What it does**: Phase flip: |1⟩ → -|1⟩
- **Matrix**: [[1, 0], [0, -1]]
- **Use**: `z_gate(qubit)`

### Hadamard Gate

**Hadamard (H)**
- **What it does**: Creates superposition
  - |0⟩ → (|0⟩ + |1⟩)/√2
  - |1⟩ → (|0⟩ - |1⟩)/√2
- **Think of it as**: Puts qubit "between" 0 and 1
- **Matrix**: (1/√2)[[1, 1], [1, -1]]
- **Use**: `hadamard(qubit)`
- **Key property**: H·H = Identity (applying twice gets you back)

## Two-Qubit Gates (NEW!)

Located in: `src/phase1_single_qubits/two_qubit_gates.py`

### CNOT (Controlled-NOT)

**Imperial Notes**: Section 2.2, Equation 35

**What it does in plain English:**
- Has two qubits: **control** and **target**
- If control = |0⟩ → do nothing
- If control = |1⟩ → flip the target

**Truth Table:**
```
|00⟩ → |00⟩  (control is 0, leave target alone)
|01⟩ → |01⟩  (control is 0, leave target alone)
|10⟩ → |11⟩  (control is 1, flip target: 0→1)
|11⟩ → |10⟩  (control is 1, flip target: 1→0)
```

**Why it matters:**
- Creates **entanglement** (makes qubits correlated)
- With single-qubit gates, can build ANY quantum circuit
- Used in: error correction, teleportation, all quantum algorithms

**Use in code:**
```python
from phase1_single_qubits.two_qubit_gates import CNOT, apply_two_qubit_gate

# Create |10⟩ state
state = np.array([0, 0, 1, 0], dtype=complex)

# Apply CNOT → becomes |11⟩
result = apply_two_qubit_gate(state, CNOT)
```

### SWAP Gate

**Imperial Notes**: Section 2.4.4, Equations 62-63

**What it does:**
Simply exchanges the two qubits!

**Examples:**
```
|01⟩ → |10⟩  (first qubit gets 1, second gets 0)
|10⟩ → |01⟩  (first qubit gets 0, second gets 1)
```

**Why it's useful:**
- Moving quantum information around
- Routing in quantum circuits
- Can be built from 3 CNOT gates

**Use in code:**
```python
from phase1_single_qubits.two_qubit_gates import SWAP

state = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
result = apply_two_qubit_gate(state, SWAP)      # → |10⟩
```

### Controlled-Z (CZ)

**Imperial Notes**: Section 2.4.1, Equation 46

**What it does:**
- Adds a minus sign if **both** qubits are |1⟩
- Otherwise does nothing

**Examples:**
```
|00⟩ → |00⟩
|01⟩ → |01⟩
|10⟩ → |10⟩
|11⟩ → -|11⟩  (gets minus sign!)
```

**Special property:**
Unlike CNOT, CZ is **symmetric** - doesn't matter which qubit is "control"!

## Relationship Between Gates

### CNOT Creates Entanglement

Starting with: |+⟩ ⊗ |0⟩ = (|0⟩ + |1⟩)/√2 ⊗ |0⟩

After CNOT:
```
(|00⟩ + |11⟩)/√2  ← Bell state! Maximally entangled!
```

This state **cannot** be written as |ψ⟩ ⊗ |φ⟩ - it's fundamentally a two-qubit phenomenon.

### Universal Gate Sets

These combinations can build ANY quantum circuit:

1. **{CNOT, All single-qubit gates}** ← Most common
2. **{CNOT, H, T}** ← Minimal universal set

## Code Examples

### Example 1: Bell State Creation
```python
from phase1_single_qubits.gates import HADAMARD
from phase1_single_qubits.two_qubit_gates import CNOT, tensor_product

# Start with |0⟩ ⊗ |0⟩
ket_0 = np.array([1, 0], dtype=complex)

# Apply H to first qubit → |+⟩ ⊗ |0⟩
ket_plus = (1/np.sqrt(2)) * np.array([1, 1], dtype=complex)

# Combine into 2-qubit state
state = tensor_product(ket_plus, ket_0)

# Apply CNOT → Bell state!
bell_state = apply_two_qubit_gate(state, CNOT)
# Result: (|00⟩ + |11⟩)/√2
```

### Example 2: Quantum Teleportation Setup
```python
# Teleportation needs a Bell pair shared between Alice and Bob
# This is the first step!

alice_qubit = ket_0
bob_qubit = ket_0

# Create entanglement between them
shared_state = tensor_product(alice_qubit, bob_qubit)
shared_state = apply_two_qubit_gate(
    apply_gate(shared_state[0], HADAMARD),
    CNOT
)
```

## From Your Imperial Notes

### Key Equations

**CNOT (Eq. 35):**
```
Û_c = |0⟩⟨0| ⊗ 𝟙 + |1⟩⟨1| ⊗ σ_x
```
Translation: "If control is 0, do identity. If control is 1, do X gate."

**SWAP (Eq. 62):**
```
SWAP |Ψ⟩|Φ⟩ = |Φ⟩|Ψ⟩
```
Translation: "Exchange the states."

**Controlled-Z (Eq. 46):**
```
Û_cz = |0⟩⟨0| ⊗ 𝟙 + |1⟩⟨1| ⊗ σ_z
```

## Testing Your Understanding

**Question 1:** What does CNOT do to |+⟩ ⊗ |0⟩?
<details>
<summary>Answer</summary>
Creates the Bell state (|00⟩ + |11⟩)/√2 - maximally entangled!
</details>

**Question 2:** Can you write SWAP using only CNOT gates?
<details>
<summary>Answer</summary>
Yes! SWAP = CNOT₁₂ · CNOT₂₁ · CNOT₁₂ (three CNOTs in sequence)
</details>

**Question 3:** What happens if you apply CNOT twice to the same state?
<details>
<summary>Answer</summary>
You get back the original state! CNOT is self-inverse: CNOT·CNOT = Identity
</details>

## Running the Demo

```bash
# See all gates in action:
python examples/two_qubit_demo.py
```

## Next Steps for Your Project

Based on the master plan, you should:

1. ✅ Implement single-qubit gates (DONE)
2. ✅ Implement two-qubit gates (DONE)
3. **Next:** Bloch sphere visualization
4. **Then:** Interactive Streamlit app for Phase 1

The two-qubit gates you now have are essential for:
- **Phase 2:** Creating Bell states (entanglement)
- **Phase 3:** Building quantum algorithms
- **Phase 5:** Quantum error correction circuits
