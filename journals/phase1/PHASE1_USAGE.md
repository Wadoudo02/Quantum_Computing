# Phase 1 Usage Guide

Complete guide for using all Phase 1 quantum computing implementations.

## Quick Start

### 1. Run the Streamlit Interactive App

The **easiest way** to explore Phase 1 functionality:

```bash
cd src/phase1_qubits
streamlit run app.py
```

This launches an interactive web app with 5 modes:
- **State Creator**: Create custom qubits with sliders
- **Gate Laboratory**: Apply gates and see transformations
- **Measurement Lab**: Simulate quantum measurements
- **Common States**: Visualize all basis states
- **Gate Sequence**: Build quantum circuits

### 2. Run the Complete Demo

Comprehensive demonstration of all Phase 1 features:

```bash
python examples/phase1_complete_demo.py
```

This runs 6 interactive demos:
1. Basic qubit creation and properties
2. Quantum gate operations
3. Quantum measurements
4. Rotation gates
5. Two-qubit systems and entanglement
6. Two-qubit gates (CNOT, SWAP, CZ)

### 3. Run Tests

Verify everything works correctly:

```bash
# Test Bloch sphere visualizer
python examples/test_bloch_sphere.py

# Test Streamlit app components
python examples/test_streamlit_app.py
```

## Core Components

### Qubit Class

Create and manipulate single qubits:

```python
from phase1_qubits.qubit import Qubit, ket_0, ket_1, ket_plus, ket_minus
import numpy as np

# Basis states
q0 = ket_0()                    # |0⟩
q1 = ket_1()                    # |1⟩
q_plus = ket_plus()             # |+⟩ = (|0⟩ + |1⟩)/√2
q_minus = ket_minus()           # |−⟩ = (|0⟩ − |1⟩)/√2

# Custom superposition
psi = Qubit([np.sqrt(0.7), np.sqrt(0.3)])

# Properties
print(psi.prob_0())              # Probability of measuring |0⟩
print(psi.prob_1())              # Probability of measuring |1⟩
print(psi.bloch_coordinates())   # (x, y, z) on Bloch sphere

# Measurement
results = psi.measure(shots=1000)
```

### Quantum Gates

Apply single-qubit gates:

```python
from phase1_qubits.gates import (
    HADAMARD, PAULI_X, PAULI_Y, PAULI_Z,
    S_GATE, T_GATE, apply_gate, rx, ry, rz
)
import numpy as np

# Pauli gates
q = ket_0()
q_flipped = apply_gate(q, PAULI_X)  # X|0⟩ = |1⟩

# Hadamard (create superposition)
q_super = apply_gate(q, HADAMARD)   # H|0⟩ = |+⟩

# Rotation gates
angle = np.pi / 4
q_rotated = apply_gate(q, rx(angle))  # Rotate around X axis

# Gate sequences
q = ket_0()
q = apply_gate(q, HADAMARD)
q = apply_gate(q, S_GATE)
q = apply_gate(q, HADAMARD)
```

### Bloch Sphere Visualization

Visualize qubits on the Bloch sphere:

```python
from phase1_qubits.bloch_sphere import BlochSphere

# Create Bloch sphere
bloch = BlochSphere(figsize=(10, 10))

# Add qubits
bloch.add_qubit(ket_0(), label="|0⟩", color='blue')
bloch.add_qubit(ket_1(), label="|1⟩", color='red')
bloch.add_qubit(ket_plus(), label="|+⟩", color='green')

# Display or save
bloch.show()                        # Interactive display
bloch.save("my_states.png")         # Save to file
```

**Helper functions:**

```python
from phase1_qubits.bloch_sphere import plot_gate_trajectory, compare_states

# Visualize gate sequence
plot_gate_trajectory(
    ket_0(),
    [HADAMARD, S_GATE, HADAMARD],
    ['H', 'S', 'H']
)

# Compare multiple states
compare_states(
    [ket_0(), ket_1(), ket_plus(), ket_minus()],
    ['|0⟩', '|1⟩', '|+⟩', '|−⟩']
)
```

### Two-Qubit Systems

Work with two-qubit states and entanglement:

```python
from phase1_qubits.multi_qubit import (
    TwoQubitSystem, tensor_product,
    bell_phi_plus, bell_phi_minus, bell_psi_plus, bell_psi_minus
)

# Create product state
q0 = ket_0()
q1 = ket_1()
system = tensor_product(q0, q1)  # |01⟩

# Check entanglement
print(system.is_entangled())         # False (product state)

# Bell states (maximally entangled)
bell = bell_phi_plus()               # (|00⟩ + |11⟩)/√2
print(bell.is_entangled())           # True
print(bell.entanglement_entropy())   # 1.0 (maximum)

# Schmidt decomposition
coeffs, basis_A, basis_B = bell.schmidt_decomposition()

# Reduced density matrices
rho_A = bell.reduced_density_matrix(0)  # Trace out qubit 1
rho_B = bell.reduced_density_matrix(1)  # Trace out qubit 0
```

### Two-Qubit Gates

Apply two-qubit gates:

```python
from phase1_qubits.two_qubit_gates import (
    CNOT, SWAP, CZ,
    apply_gate_to_system, apply_single_qubit_gate
)

# Create entanglement with CNOT
system = tensor_product(ket_0(), ket_0())   # |00⟩
system = apply_single_qubit_gate(system, HADAMARD, qubit_index=0)
system = apply_gate_to_system(system, CNOT)  # Creates |Φ+⟩ Bell state

# SWAP gate
system = tensor_product(ket_0(), ket_1())   # |01⟩
system = apply_gate_to_system(system, SWAP) # |10⟩

# Controlled-Z
system = apply_gate_to_system(system, CZ)
```

## Example Scripts

### Quick Example: Create and Visualize

```python
from phase1_qubits.qubit import Qubit
from phase1_qubits.gates import HADAMARD, apply_gate
from phase1_qubits.bloch_sphere import BlochSphere
import numpy as np

# Create custom state
theta = np.pi / 3  # 60 degrees
phi = np.pi / 4    # 45 degrees

alpha = np.cos(theta / 2)
beta = np.exp(1j * phi) * np.sin(theta / 2)
psi = Qubit([alpha, beta])

# Apply Hadamard
psi_after = apply_gate(psi, HADAMARD)

# Visualize before and after
bloch = BlochSphere()
bloch.add_qubit(psi, label="Before", color='blue')
bloch.add_qubit(psi_after, label="After H", color='red')
bloch.show()
```

### Quick Example: Measurement Statistics

```python
from phase1_qubits.qubit import ket_plus
import numpy as np
import matplotlib.pyplot as plt

# Create superposition
q = ket_plus()  # Equal superposition

# Measure many times
results = q.measure(shots=1000)

# Plot histogram
plt.hist(results, bins=[0, 1, 2], align='left', rwidth=0.8)
plt.xlabel('Measurement Outcome')
plt.ylabel('Counts')
plt.title('Measurement of |+⟩ state')
plt.xticks([0, 1], ['|0⟩', '|1⟩'])
plt.show()
```

### Quick Example: Create Bell State

```python
from phase1_qubits.qubit import ket_0
from phase1_qubits.gates import HADAMARD
from phase1_qubits.multi_qubit import tensor_product
from phase1_qubits.two_qubit_gates import CNOT, apply_gate_to_system, apply_single_qubit_gate

# Start with |00⟩
system = tensor_product(ket_0(), ket_0())

# Apply H ⊗ I
system = apply_single_qubit_gate(system, HADAMARD, qubit_index=0)

# Apply CNOT
bell_state = apply_gate_to_system(system, CNOT)

# Verify entanglement
print(f"Is entangled? {bell_state.is_entangled()}")
print(f"Entanglement entropy: {bell_state.entanglement_entropy():.4f}")
print(f"State: {bell_state.state}")
```

## File Structure

```
src/phase1_qubits/
├── qubit.py              # Qubit class and basis states
├── gates.py              # Single-qubit gates
├── bloch_sphere.py       # 3D Bloch sphere visualization
├── multi_qubit.py        # Two-qubit systems
├── two_qubit_gates.py    # Two-qubit gates (CNOT, SWAP, CZ)
└── app.py                # Streamlit interactive app

examples/
├── phase1_complete_demo.py        # Complete Phase 1 demo
├── test_bloch_sphere.py           # Test visualizer
├── test_streamlit_app.py          # Test app components
├── two_qubit_demo.py              # Two-qubit examples
└── architecture_demo.py           # Architecture overview

docs/
├── architecture.md                # Code architecture
└── theory/
    └── gates_summary.md           # Gate reference
```

## Common Tasks

### Task: Create a custom qubit and measure it

```python
from phase1_qubits.qubit import Qubit
import numpy as np

# 70% chance of |0⟩, 30% chance of |1⟩
psi = Qubit([np.sqrt(0.7), np.sqrt(0.3)])

# Single measurement
outcome = psi.measure(shots=1)[0]
print(f"Measured: {outcome}")

# Many measurements for statistics
results = psi.measure(shots=1000)
print(f"P(0) ≈ {np.sum(results == 0) / 1000:.3f}")
```

### Task: Apply a sequence of gates

```python
from phase1_qubits.qubit import ket_0
from phase1_qubits.gates import HADAMARD, S_GATE, T_GATE, apply_gate

q = ket_0()
for gate in [HADAMARD, S_GATE, T_GATE]:
    q = apply_gate(q, gate)
    print(q)
```

### Task: Visualize gate trajectory

```python
from phase1_qubits.qubit import ket_0
from phase1_qubits.gates import HADAMARD, PAULI_X, PAULI_Z
from phase1_qubits.bloch_sphere import plot_gate_trajectory

plot_gate_trajectory(
    initial_qubit=ket_0(),
    gates=[HADAMARD, PAULI_Z, PAULI_X],
    gate_names=['H', 'Z', 'X'],
    title="My Gate Sequence"
)
```

### Task: Check if a two-qubit state is entangled

```python
from phase1_qubits.multi_qubit import TwoQubitSystem

# Product state (not entangled)
state1 = TwoQubitSystem([1, 0, 0, 0])  # |00⟩
print(state1.is_entangled())  # False

# Bell state (entangled)
state2 = TwoQubitSystem([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])  # |Φ+⟩
print(state2.is_entangled())  # True
```

## Troubleshooting

### Import errors

Make sure you're in the project root:
```bash
cd /path/to/Quantum_Computing
python examples/phase1_complete_demo.py
```

Or add src to your Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

### Visualization not showing

For Bloch sphere visualization, use `.save()` instead of `.show()`:
```python
bloch.save("my_plot.png")
```

### Streamlit app won't start

Install Streamlit:
```bash
pip install streamlit
```

Check dependencies:
```bash
pip install numpy matplotlib scipy
```

## Next Steps

After completing Phase 1:

1. ✅ Review all Phase 1 deliverables in `quantum_master_plan.md`
2. Create Jupyter notebook with explanations
3. Add unit tests for all functionality
4. Move to **Phase 2**: Entanglement and Bell States
5. Prepare blog post: "Understanding Qubits Through Visualization"

## Resources

- **Imperial College Notes**: Sections 1.1, 1.4, 2.2, 2.4
- **Project Plan**: `quantum_master_plan.md`
- **Architecture**: `docs/architecture.md`
- **Theory Reference**: `docs/theory/gates_summary.md`

## For Recruiters

**Best demonstrations for Quantinuum/Riverlane:**

1. **Interactive Demo**: Run `streamlit run src/phase1_qubits/app.py`
2. **Complete Walkthrough**: Run `python examples/phase1_complete_demo.py`
3. **Code Review**: Start with `src/phase1_qubits/qubit.py`

**Key achievements:**
- ✅ Complete qubit implementation with Bloch sphere
- ✅ All single-qubit gates from Imperial notes
- ✅ Two-qubit systems and entanglement
- ✅ Interactive visualization tools
- ✅ Comprehensive testing and documentation
