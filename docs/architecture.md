# Code Architecture: Qubit Systems

## Overview

The quantum computing framework is organized to handle single qubits, two-qubit systems, and eventually N-qubit systems.

## File Structure

```
src/phase1_qubits/
├── qubit.py              # Single qubit class
├── multi_qubit.py        # Multi-qubit systems (TwoQubitSystem, etc.)
├── gates.py              # Single-qubit gates (X, Y, Z, H, etc.)
└── two_qubit_gates.py    # Two-qubit gates (CNOT, SWAP, CZ, etc.)
```

## Class Hierarchy

### 1. Single Qubit (`qubit.py`)

**Class: `Qubit`**
- Represents a single qubit: |ψ⟩ = α|0⟩ + β|1⟩
- State vector: 2D complex array
- Methods:
  - `measure()` - Measure in computational basis
  - `prob_0()`, `prob_1()` - Calculate probabilities
  - `bloch_coordinates()` - Get Bloch sphere position
  - `copy()` - Create a copy

**Convenience functions:**
```python
from phase1_qubits.qubit import ket_0, ket_1, ket_plus, ket_minus

q0 = ket_0()      # |0⟩
q1 = ket_1()      # |1⟩
qp = ket_plus()   # (|0⟩+|1⟩)/√2
qm = ket_minus()  # (|0⟩-|1⟩)/√2
```

### 2. Two-Qubit System (`multi_qubit.py`)

**Class: `TwoQubitSystem`**
- Represents two qubits: |ψ⟩ = c₀₀|00⟩ + c₀₁|01⟩ + c₁₀|10⟩ + c₁₁|11⟩
- State vector: 4D complex array [c₀₀, c₀₁, c₁₀, c₁₁]
- Methods:
  - `from_single_qubits(q1, q2)` - Create from tensor product
  - `measure()` - Measure both qubits
  - `measure_qubit(index)` - Measure one qubit
  - `is_entangled()` - Check if entangled
  - `schmidt_decomposition()` - Get Schmidt coefficients
  - `entanglement_entropy()` - Calculate von Neumann entropy
  - `reduced_density_matrix(index)` - Trace out one qubit

**Convenience functions:**
```python
from phase1_qubits.multi_qubit import (
    two_ket_00, two_ket_01, two_ket_10, two_ket_11,
    bell_phi_plus, bell_phi_minus,
    bell_psi_plus, bell_psi_minus
)

# Basis states
sys = two_ket_01()  # |01⟩

# Bell states (maximally entangled)
bell = bell_phi_plus()  # (|00⟩+|11⟩)/√2
```

### 3. Single-Qubit Gates (`gates.py`)

**Matrix constants:**
- `PAULI_X` (NOT gate)
- `PAULI_Y`
- `PAULI_Z` (phase flip)
- `HADAMARD` (superposition creator)
- `S_GATE`, `T_GATE` (phase gates)
- `IDENTITY`

**Functions:**
```python
from phase1_qubits.gates import x_gate, hadamard, apply_gate

q = ket_0()
q = hadamard(q)     # Create superposition
q = x_gate(q)       # Apply NOT gate
```

**Rotation gates:**
```python
from phase1_qubits.gates import rotation_x, rotation_y, rotation_z

gate = rotation_x(np.pi/4)  # Rotate π/4 around X axis
q = apply_gate(q, gate)
```

### 4. Two-Qubit Gates (`two_qubit_gates.py`)

**Matrix constants:**
- `CNOT` - Controlled-NOT
- `SWAP` - Exchange qubits
- `CZ` - Controlled-Z
- `SQRT_SWAP` - Square root of SWAP

**Functions:**
```python
from phase1_qubits.two_qubit_gates import CNOT, SWAP, apply_gate_to_system

# Option 1: Work with TwoQubitSystem objects
from phase1_qubits.multi_qubit import two_ket_10

sys = two_ket_10()                          # |10⟩
sys = apply_gate_to_system(sys, CNOT)       # → |11⟩

# Option 2: Work with raw state vectors
import numpy as np
from phase1_qubits.two_qubit_gates import apply_two_qubit_gate

state = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
result = apply_two_qubit_gate(state, CNOT)      # → |11⟩
```

**General controlled gates:**
```python
from phase1_qubits.two_qubit_gates import controlled_u, controlled_phase
from phase1_qubits.gates import HADAMARD

# Controlled-Hadamard
ch_gate = controlled_u(HADAMARD)

# Controlled-Phase with custom angle
cp_gate = controlled_phase(np.pi/4)
```

## Usage Examples

### Example 1: Single Qubit Operations

```python
from phase1_qubits.qubit import ket_0
from phase1_qubits.gates import hadamard, x_gate

# Start with |0⟩
q = ket_0()
print(q)  # |0⟩

# Create superposition
q = hadamard(q)
print(q)  # 0.71|0⟩ + 0.71|1⟩

# Measure (probabilistic)
results = q.measure(shots=100)
print(f"0s: {sum(results == 0)}, 1s: {sum(results == 1)}")
# Output: roughly 50/50
```

### Example 2: Creating Entanglement

```python
from phase1_qubits.qubit import ket_0
from phase1_qubits.multi_qubit import TwoQubitSystem
from phase1_qubits.gates import hadamard
from phase1_qubits.two_qubit_gates import CNOT, apply_gate_to_system

# Start with |0⟩ ⊗ |0⟩
q1 = hadamard(ket_0())  # Create |+⟩
q2 = ket_0()

# Combine into two-qubit system
sys = TwoQubitSystem.from_single_qubits(q1, q2)
print(sys)  # 0.707|00⟩ + 0.707|10⟩

# Apply CNOT → Bell state!
sys = apply_gate_to_system(sys, CNOT)
print(sys)  # 0.707|00⟩ + 0.707|11⟩

# Check if entangled
print(sys.is_entangled())  # True
print(f"Entanglement entropy: {sys.entanglement_entropy():.3f}")  # 1.000
```

### Example 3: Schmidt Decomposition

```python
from phase1_qubits.multi_qubit import bell_phi_plus

# Bell state
bell = bell_phi_plus()

# Get Schmidt decomposition
coeffs, basis_A, basis_B = bell.schmidt_decomposition()

print(f"Schmidt coefficients: {coeffs}")
# [0.707, 0.707] - two equal coefficients means maximally entangled

print(f"Schmidt rank: {len(coeffs)}")  # 2
```

### Example 4: Quantum Circuit Building

```python
from phase1_qubits.qubit import ket_0
from phase1_qubits.multi_qubit import TwoQubitSystem
from phase1_qubits.gates import hadamard, x_gate
from phase1_qubits.two_qubit_gates import CNOT, apply_gate_to_system

# Quantum teleportation setup (Bell pair creation)
def create_bell_pair():
    """Create entangled Bell pair for teleportation."""
    alice = hadamard(ket_0())
    bob = ket_0()

    pair = TwoQubitSystem.from_single_qubits(alice, bob)
    pair = apply_gate_to_system(pair, CNOT)

    return pair

bell_pair = create_bell_pair()
print(f"Entangled: {bell_pair.is_entangled()}")  # True
```

## Design Philosophy

### Why Separate Classes?

1. **`Qubit` for single qubits:**
   - Simple, focused interface
   - Bloch sphere visualization
   - Easy to understand for beginners

2. **`TwoQubitSystem` for two qubits:**
   - Handles entanglement
   - Schmidt decomposition
   - Partial measurement
   - Foundation for Phase 2 (Entanglement)

3. **Future: `MultiQubitSystem` for N qubits:**
   - Will be needed for Phase 3 (Algorithms)
   - General N-qubit operations
   - Quantum circuits

### Why Separate Gate Files?

1. **`gates.py` - Single-qubit gates:**
   - 2×2 matrices
   - Phase 1 complete implementation
   - Visualizable on Bloch sphere

2. **`two_qubit_gates.py` - Two-qubit gates:**
   - 4×4 matrices
   - Creates entanglement
   - Building blocks for algorithms

## Roadmap

### Phase 1 (Current) ✅
- [x] Single `Qubit` class
- [x] Single-qubit gates (X, Y, Z, H)
- [x] `TwoQubitSystem` class
- [x] Two-qubit gates (CNOT, SWAP, CZ)
- [ ] Bloch sphere visualization
- [ ] Streamlit interactive app

### Phase 2 (Entanglement)
- [ ] Bell state demonstrations
- [ ] Entanglement measures
- [ ] Partial trace visualization
- [ ] Use `TwoQubitSystem` extensively

### Phase 3 (Algorithms)
- [ ] `MultiQubitSystem` class (N qubits)
- [ ] Quantum circuits
- [ ] Deutsch-Jozsa algorithm
- [ ] Grover's algorithm

### Phase 5 (Error Correction)
- [ ] Three-qubit operations (error correction codes)
- [ ] Stabilizer measurements
- [ ] Syndrome detection

## Testing

Each file has corresponding tests:

```bash
# Test single qubits
pytest tests/test_phase1.py::TestQubit

# Test multi-qubit systems
pytest tests/test_phase1.py::TestTwoQubitSystem

# Test gates
pytest tests/test_phase1.py::TestGates
```

## Best Practices

1. **Use classes for state representation:**
   ```python
   sys = TwoQubitSystem(...)  # Good
   state = np.array([...])    # Only for low-level work
   ```

2. **Use convenience functions:**
   ```python
   bell = bell_phi_plus()     # Good
   bell = TwoQubitSystem([1/sqrt(2), 0, 0, 1/sqrt(2)])  # Verbose
   ```

3. **Check entanglement:**
   ```python
   if sys.is_entangled():
       print("Quantum correlations present!")
   ```

4. **Use proper gate application:**
   ```python
   # For TwoQubitSystem
   sys = apply_gate_to_system(sys, CNOT)

   # For raw vectors (advanced)
   state = apply_two_qubit_gate(state, CNOT)
   ```

## Common Pitfalls

### ❌ Wrong: Mixing dimensions
```python
from phase1_qubits.qubit import Qubit
from phase1_qubits.two_qubit_gates import CNOT

q = Qubit([1, 0])
# Can't apply CNOT to single qubit!
# result = apply_gate(q, CNOT)  # ERROR: dimension mismatch
```

### ✅ Right: Use appropriate classes
```python
from phase1_qubits.multi_qubit import TwoQubitSystem
from phase1_qubits.two_qubit_gates import CNOT, apply_gate_to_system

sys = TwoQubitSystem([1, 0, 0, 0])
sys = apply_gate_to_system(sys, CNOT)  # Works!
```

### ❌ Wrong: Forgetting to normalize
```python
q = Qubit([1, 2])  # Auto-normalizes by default
q = Qubit([1, 2], normalize=False)  # Might not be normalized!
```

### ✅ Right: Check normalization
```python
q = Qubit([1, 2])
print(q.is_normalized())  # True
```

## Summary

The architecture separates concerns:

| File | Purpose | Matrix Size | Phase |
|------|---------|-------------|-------|
| `qubit.py` | Single qubits | 2D state | 1 |
| `gates.py` | Single-qubit gates | 2×2 | 1 |
| `multi_qubit.py` | Multi-qubit systems | 4D+ state | 1-2 |
| `two_qubit_gates.py` | Two-qubit gates | 4×4 | 1-2 |

This structure makes it easy to:
1. Learn progressively (single → two → N qubits)
2. Build quantum circuits
3. Implement algorithms (Phases 3-6)
4. Understand entanglement (Phase 2)
