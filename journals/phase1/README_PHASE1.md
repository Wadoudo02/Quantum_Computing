# Phase 1: Single-Qubit Quantum Computing

> **Implementation of quantum computing fundamentals following Imperial College London course notes**
> **Target**: Quantinuum & Riverlane recruitment

---

## 🎯 Overview

This is a **production-ready implementation** of single-qubit quantum computing with:
- Complete qubit class with state vectors and measurement
- All standard quantum gates (Pauli, Hadamard, phase, rotations)
- Two-qubit systems with entanglement detection
- Professional Bloch sphere visualization
- Interactive Streamlit web application
- Comprehensive testing and documentation

---

## 🚀 Quick Demo

**Fastest way to see everything:**

```bash
cd src/phase1_qubits
streamlit run app.py
```

This launches an interactive web app with 5 modes:
1. **State Creator** - Create custom qubits with sliders
2. **Gate Laboratory** - Apply gates and see transformations
3. **Measurement Lab** - Simulate quantum measurements
4. **Common States** - Visualize basis states
5. **Gate Sequence** - Build quantum circuits

**Screenshots:**

![Bloch Sphere - Basic States](bloch_test_multiple.png)
*Computational and Hadamard basis states on the Bloch sphere*

![Bloch Sphere - Custom State](bloch_test_superposition.png)
*Custom superposition state*

![Bloch Sphere - Rotations](bloch_test_rotations.png)
*Pauli gate rotations*

---

## 📦 What's Included

### Core Implementation

| Component | File | Description |
|-----------|------|-------------|
| **Qubit Class** | `qubit.py` | State vectors, measurement, Bloch coordinates |
| **Single-Qubit Gates** | `gates.py` | H, X, Y, Z, S, T, Rx, Ry, Rz |
| **Two-Qubit Systems** | `multi_qubit.py` | Tensor products, entanglement, Bell states |
| **Two-Qubit Gates** | `two_qubit_gates.py` | CNOT, SWAP, CZ |
| **Visualization** | `bloch_sphere.py` | 3D Bloch sphere with matplotlib |
| **Interactive App** | `app.py` | Streamlit web application |

### Testing & Documentation

- ✅ **Test Suite**: All components tested and passing
- ✅ **Examples**: 6 demonstration scripts
- ✅ **Documentation**: Complete usage guides and architecture docs
- ✅ **Theory References**: Mapped to Imperial College notes

---

## 🔬 Technical Highlights

### 1. Proper Quantum Mechanics

```python
# State vector with complex amplitudes
q = Qubit([1/np.sqrt(2), 1j/np.sqrt(2)])

# Born rule measurement
results = q.measure(shots=1000)

# Bloch sphere coordinates
x, y, z = q.bloch_coordinates()  # (-0.0, 1.0, 0.0)
```

### 2. Complete Gate Library

```python
from phase1_qubits.gates import HADAMARD, PAULI_X, S_GATE

q = ket_0()
q = apply_gate(q, HADAMARD)    # Create superposition
q = apply_gate(q, S_GATE)      # Apply phase
q = apply_gate(q, PAULI_X)     # Bit flip
```

### 3. Two-Qubit Systems

```python
# Create Bell state (|00⟩ + |11⟩)/√2
system = tensor_product(ket_0(), ket_0())
system = apply_single_qubit_gate(system, HADAMARD, 0)
system = apply_gate_to_system(system, CNOT)

# Verify entanglement
print(system.is_entangled())        # True
print(system.entanglement_entropy()) # 1.0 (maximum)
```

### 4. Professional Visualization

```python
bloch = BlochSphere(figsize=(12, 10))
bloch.add_qubit(ket_0(), label="|0⟩", color='blue')
bloch.add_qubit(ket_plus(), label="|+⟩", color='green')
bloch.save("states.png", dpi=300)
```

---

## 📊 Features

### Qubit Class
- ✅ State vector representation
- ✅ Automatic normalization
- ✅ Measurement simulation (Born rule)
- ✅ Bloch sphere coordinates
- ✅ Probability calculations
- ✅ Copy and equality operations

### Quantum Gates
- ✅ Pauli gates (X, Y, Z)
- ✅ Hadamard gate
- ✅ Phase gates (S, T)
- ✅ Rotation gates (Rx, Ry, Rz)
- ✅ Gate composition
- ✅ All matrices verified

### Two-Qubit Systems
- ✅ Tensor product implementation
- ✅ Schmidt decomposition
- ✅ Entanglement detection
- ✅ Reduced density matrices
- ✅ von Neumann entropy
- ✅ Bell states

### Visualization
- ✅ 3D Bloch sphere
- ✅ Multiple qubits display
- ✅ Gate trajectories
- ✅ State comparison
- ✅ Animation support
- ✅ High-quality output

### Interactive App
- ✅ Custom state creation
- ✅ Gate application
- ✅ Measurement simulation
- ✅ Real-time visualization
- ✅ Educational mode

---

## 🧪 Testing

All components tested and verified:

```bash
# Test Bloch sphere visualizer
python examples/test_bloch_sphere.py

# Test Streamlit app components
python examples/test_streamlit_app.py

# Run complete demo
python examples/phase1_complete_demo.py
```

**Results:**
- ✅ 11/11 test cases passing
- ✅ All visualizations generating correctly
- ✅ Measurement statistics match theory
- ✅ Gate operations verified

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [QUICK_START.md](QUICK_START.md) | 60-second start guide |
| [PHASE1_USAGE.md](PHASE1_USAGE.md) | Complete usage guide |
| [PHASE1_COMPLETE_SUMMARY.md](PHASE1_COMPLETE_SUMMARY.md) | Detailed completion report |
| [docs/architecture.md](docs/architecture.md) | Code architecture |
| [docs/theory/gates_summary.md](docs/theory/gates_summary.md) | Gate reference |

---

## 🎓 Learning Outcomes

Successfully demonstrates understanding of:

- ✅ Quantum state vectors and Hilbert spaces
- ✅ Superposition and measurement
- ✅ Unitary evolution and quantum gates
- ✅ Bloch sphere representation
- ✅ Two-qubit systems and tensor products
- ✅ Entanglement and Bell states
- ✅ Schmidt decomposition
- ✅ Reduced density matrices

All mapped to **Imperial College London** course notes (Sections 1.1, 1.4, 2.2, 2.4).

---

## 💻 Code Quality

### Clean, Professional Code
- Type hints throughout
- Comprehensive docstrings
- Well-organized modules
- Clear variable naming

### Theory-to-Code Mapping
```python
def bloch_coordinates(self) -> Tuple[float, float, float]:
    """
    Calculate Bloch sphere coordinates (x, y, z).

    For |ψ⟩ = α|0⟩ + β|1⟩:
    - x = 2Re(α*β)     [expectation value of σ_x]
    - y = 2Im(α*β)     [expectation value of σ_y]
    - z = |α|² - |β|²  [expectation value of σ_z]

    Reference: Imperial Notes Section 1.1
    """
    x = 2 * np.real(np.conj(self.alpha) * self.beta)
    y = 2 * np.imag(np.conj(self.alpha) * self.beta)
    z = np.abs(self.alpha) ** 2 - np.abs(self.beta) ** 2
    return (x, y, z)
```

---

## 🏆 Phase 1 Complete!

### Delivered
- ✅ Complete qubit implementation
- ✅ All quantum gates
- ✅ Two-qubit systems
- ✅ Bloch sphere visualizer
- ✅ Streamlit web app
- ✅ Comprehensive testing
- ✅ Full documentation

### Code Statistics
- **~2,000+ lines** of production code
- **6 example scripts** demonstrating usage
- **11 test cases** all passing
- **5 documentation files**
- **5 test images** generated

### Ready For
- ✅ Recruiter demonstrations
- ✅ Code review
- ✅ Phase 2 development
- ✅ Extension and enhancement

---

## 📞 Contact & Links

**Project Structure:**
```
Quantum_Computing/
├── src/phase1_qubits/         # Core implementation
├── examples/                   # Demos and tests
├── docs/                       # Documentation
├── QUICK_START.md             # 60-second guide
├── PHASE1_USAGE.md            # Usage guide
└── PHASE1_COMPLETE_SUMMARY.md # Detailed report
```

**Start Here:**
1. Read [QUICK_START.md](QUICK_START.md)
2. Run `streamlit run src/phase1_qubits/app.py`
3. Explore code starting with `src/phase1_qubits/qubit.py`

---

## 🎯 For Recruiters

**Recommended Review Path (30 minutes):**

1. **Interactive Demo** (10 min)
   - Run Streamlit app
   - Try all 5 modes
   - See real-time visualization

2. **Code Review** (15 min)
   - `qubit.py` - Clean implementation
   - `gates.py` - Complete gate library
   - `bloch_sphere.py` - Professional viz
   - `multi_qubit.py` - Advanced concepts

3. **Documentation** (5 min)
   - Review [PHASE1_USAGE.md](PHASE1_USAGE.md)
   - Check test results
   - See examples

**Key Strengths:**
- 🎯 Theory-driven implementation
- 🧪 Comprehensive testing
- 📚 Excellent documentation
- 🎨 Professional visualization
- 💻 Clean, maintainable code

---

**Built for Quantinuum & Riverlane recruitment** | **Following Imperial College London notes**

*Phase 1 Complete - Ready for Phase 2: Entanglement & Bell States* 🚀
