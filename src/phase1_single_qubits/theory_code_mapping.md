# Phase 1: Theory-to-Code Mapping Guide

This guide maps the code you've written to specific sections in your Imperial College Quantum Information notes.

---

## 📖 Reading Order & Code Connection

### **1. Qubit Class (`qubit.py`)**

#### **Section 1.1.1: States and Operators (Pages 3-4)**

**Read First:** Equations (1)-(6)

**Key Concepts:**
- **Equation (2):** The quantum state representation
  ```
  |ψ⟩ = α|0⟩ + β|1⟩
  ```
  **In your code:** This is `self.state = [α, β]` in the `Qubit.__init__()` method

- **Basis vectors:**
  ```
  |0⟩ = [1, 0]ᵀ    |1⟩ = [0, 1]ᵀ
  ```
  **In your code:** The `ket_0()` and `ket_1()` functions

- **Normalisation condition:** |α|² + |β|² = 1
  **In your code:** The `_normalize()` method and `is_normalized()` check

**What to understand:**
- Why qubits are 2D complex vectors
- Why normalisation is necessary (probabilities must sum to 1)
- The difference between α (amplitude) and |α|² (probability)

---

#### **Section 1.1.3: Measurement (Page 4)**

**Read:** Equations (13)-(15)

**Key Concepts:**
- **Projectors:** P̂₀ = |0⟩⟨0|, P̂₁ = |1⟩⟨1|
- **Born Rule:** pᵢ = ⟨ψ|P̂ᵢ|ψ⟩ = |αᵢ|²
  
**In your code:**
```python
def prob_0(self) -> float:
    return np.abs(self.alpha) ** 2  # This is |α|²

def measure(self, shots: int = 1):
    probabilities = [self.prob_0(), self.prob_1()]
    outcomes = np.random.choice([0, 1], size=shots, p=probabilities)
```

**Exercise for you:**
1. Create a qubit in superposition: `q = Qubit([1/√2, 1/√2])`
2. Measure it 1000 times: `results = q.measure(1000)`
3. Count outcomes: Should be roughly 50% zeros, 50% ones
4. **Why?** Because |1/√2|² = 0.5 for both states

---

#### **Bloch Sphere Representation**

**Read:** The notes mention Pauli matrices (Equation 5) and their geometric meaning

**Key Theory:** Any single qubit can be written as:
```
|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
```

Where:
- θ ∈ [0, π] is the polar angle (latitude)
- φ ∈ [0, 2π) is the azimuthal angle (longitude)

**In your code:**
```python
def bloch_coordinates(self) -> Tuple[float, float, float]:
    # Calculates (x, y, z) on unit sphere
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)  
    z = np.abs(alpha) ** 2 - np.abs(beta) ** 2
```

**This comes from:** Expectation values of Pauli operators
- x = ⟨σₓ⟩
- y = ⟨σᵧ⟩  
- z = ⟨σᵤ⟩

**Try this:**
```python
from phase1_single_qubits.qubit import *

# North pole (|0⟩)
q0 = ket_0()
print(f"|0⟩ is at: {q0.bloch_coordinates()}")  # Should be (0, 0, 1)

# South pole (|1⟩)
q1 = ket_1()
print(f"|1⟩ is at: {q1.bloch_coordinates()}")  # Should be (0, 0, -1)

# Equator (|+⟩)
qplus = ket_plus()
print(f"|+⟩ is at: {qplus.bloch_coordinates()}")  # Should be (1, 0, 0)
```

---

### **2. Gates Module (`gates.py`)**

#### **Section 1.1.1: Pauli Matrices (Page 3, Equation 5)**

**Read carefully:**
```
σₓ = [0 1]    σᵧ = [0  -i]    σᵤ = [1   0]
     [1 0]         [i   0]         [0  -1]
```

**In your code:**
```python
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
```

**Key properties from Equation (6):**
- **Commutator:** [σₐ, σᵦ] = 2iεₐᵦᵧσᵧ
- **Anti-commutator:** {σₐ, σᵦ} = δₐᵦ𝟙

**What this means:**
- Pauli matrices don't commute (order matters!)
- They square to identity: σₓ² = σᵧ² = σᵤ² = 𝟙

**Test this:**
```python
import numpy as np
from phase1_single_qubits.gates import *

# Verify σₓ² = I
result = PAULI_X @ PAULI_X
print(np.allclose(result, IDENTITY))  # Should be True

# Check they don't commute
XY = PAULI_X @ PAULI_Y
YX = PAULI_Y @ PAULI_X
print(np.allclose(XY, YX))  # Should be False!
```

---

#### **Section 2.2: Hadamard Gate (Page 8, Equation 33)**

**Read:** 
```
H = 1/√2 [1   1]
         [1  -1]
```

**In your code:**
```python
HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
```

**Critical property:** H² = I (Hadamard is its own inverse!)

**What Hadamard does:**
- |0⟩ → |+⟩ = (|0⟩ + |1⟩)/√2
- |1⟩ → |−⟩ = (|0⟩ − |1⟩)/√2
- |+⟩ → |0⟩ (inverse operation!)

**Try the circuit from your notes (Figure 2a):**
```python
from phase1_single_qubits.qubit import ket_0
from phase1_single_qubits.gates import hadamard, rotation_z

# Mach-Zehnder interferometer (Equation 29)
q = ket_0()
q = hadamard(q)           # First H gate
q = apply_gate(q, rotation_z(phi))  # Phase shift
q = hadamard(q)           # Second H gate

# Measure probability
print(f"Probability at |0⟩: {q.prob_0()}")  # Varies with phi!
```

---

#### **Section 1.1.2: Time Evolution (Page 3-4, Equations 7-12)**

**Read:** The Schrödinger equation
```
iℏ d/dt|Ψ(t)⟩ = Ĥ|Ψ(t)⟩
```

**Solution (Equation 8):**
```
|Ψ(t)⟩ = Û(t,t₀)|Ψ(t₀)⟩ = e^(-iĤ(t-t₀))|Ψ(t₀)⟩
```

**In your code:**
```python
def time_evolution(qubit: Qubit, hamiltonian: np.ndarray, time: float) -> Qubit:
    evolution_operator = np.array(expm(-1j * hamiltonian * time))
    return apply_gate(qubit, evolution_operator)
```

**Key insight:** Gates ARE time evolution!
- A gate U is just e^(-iĤt) for some H and t
- Gates are unitary because time evolution preserves normalisation

**Example from notes:**
```python
# Evolve |+⟩ under σᵤ Hamiltonian (like Ramsey scheme)
q = ket_plus()
H = PAULI_Z
omega = 1.0  # frequency

# Equation (101) from your notes
for t in [0, np.pi/4, np.pi/2, np.pi]:
    q_evolved = time_evolution(q, omega * H, t)
    print(f"t={t:.2f}: P(|+⟩) = {0.5 * (1 + np.cos(omega*t)):.3f}")
```

---

#### **Rotation Gates - Bloch Sphere Intuition**

**Not explicitly in notes, but fundamental:**

Any single-qubit unitary can be written as:
```
U = e^(-iθ/2 n·σ) = cos(θ/2)I - i sin(θ/2)(nₓσₓ + nᵧσᵧ + nᵤσᵤ)
```

This rotates the Bloch sphere by angle θ around axis n.

**In your code:**
```python
def rotation_x(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
```

**Note the θ/2:** Qubit rotations are "double covering" of physical rotations!

**Relationship to Pauli gates:**
- X = Rₓ(π) - 180° rotation around X
- Y = Rᵧ(π) - 180° rotation around Y  
- Z = Rᵤ(π) - 180° rotation around Z

**Verify:**
```python
# X gate should equal Rₓ(π)
Rx_pi = rotation_x(np.pi)
print(np.allclose(Rx_pi, PAULI_X))  # True (up to global phase)
```

---

## 🎯 Learning Exercises

### **Exercise 1: Understanding Superposition**
```python
# Create superposition
q = ket_0()
q = hadamard(q)  # Now in |+⟩ = (|0⟩ + |1⟩)/√2

# Measure many times
results = q.measure(1000)
print(f"Measured 0: {np.sum(results == 0)} times")
print(f"Measured 1: {np.sum(results == 1)} times")
```

**Question:** Before measuring, where was the qubit? (Answer: In both states simultaneously!)

---

### **Exercise 2: Gate Sequences**
```python
# What does H-X-H do?
q = ket_0()
q = hadamard(q)
q = x_gate(q)
q = hadamard(q)
print(q)  # What state is this?

# Try to predict before running!
```

**Answer:** This implements a Z gate! (H-X-H = Z)

---

### **Exercise 3: Verifying Unitarity**

From Equation (10), gates must be unitary: U†U = I

```python
from phase1_single_qubits.gates import is_unitary

# All these should be True
print(is_unitary(PAULI_X))
print(is_unitary(HADAMARD))
print(is_unitary(rotation_y(0.7)))

# What about non-unitary?
not_unitary = np.array([[1, 0], [0, 2]])  # Not unitary!
print(is_unitary(not_unitary))  # False
```

**Why must gates be unitary?** They preserve normalisation (probability = 1)

---

### **Exercise 4: Bloch Sphere Visualization**

```python
# Create qubits at different Bloch sphere positions
qubits = {
    "|0⟩": ket_0(),
    "|1⟩": ket_1(),
    "|+⟩": ket_plus(),
    "|−⟩": ket_minus(),
    "random": random_qubit()
}

for name, q in qubits.items():
    x, y, z = q.bloch_coordinates()
    theta, phi = q.bloch_angles()
    print(f"{name:8s}: (x,y,z)=({x:.2f}, {y:.2f}, {z:.2f}), "
          f"(θ,φ)=({theta:.2f}, {phi:.2f})")
```

**Understanding:**
- |0⟩ at north pole: z = +1
- |1⟩ at south pole: z = -1
- |+⟩ on equator: x = +1
- |−⟩ on equator: x = -1

---

## 📝 Before Moving to Visualization

### **Check Your Understanding:**

1. **Can you explain in words:**
   - What a qubit is physically? (Two-level quantum system)
   - Why amplitudes are complex? (Phase matters for interference!)
   - What measurement does? (Projects onto basis, gives probabilistic outcome)

2. **Can you calculate by hand:**
   - Apply X gate to |0⟩ → ? (Should get |1⟩)
   - Apply H gate to |0⟩ → ? (Should get (|0⟩+|1⟩)/√2)
   - Probability of measuring |0⟩ from |+⟩? (Should get 1/2)

3. **Can you verify in code:**
   - Create |+⟩ and measure it 100 times
   - Apply H twice to |0⟩ and verify you get |0⟩ back
   - Check that all Pauli matrices square to identity

---

## 🔄 Summary: Notes to Code

| Imperial Notes | Code File | Key Functions |
|----------------|-----------|---------------|
| Section 1.1.1 (States) | `qubit.py` | `Qubit.__init__`, `ket_0()`, `ket_1()` |
| Section 1.1.3 (Measurement) | `qubit.py` | `measure()`, `prob_0()`, `prob_1()` |
| Equation 5 (Pauli matrices) | `gates.py` | `PAULI_X`, `PAULI_Y`, `PAULI_Z` |
| Equation 33 (Hadamard) | `gates.py` | `HADAMARD`, `hadamard()` |
| Section 1.1.2 (Dynamics) | `gates.py` | `time_evolution()`, `apply_gate()` |
| Bloch sphere (implicit) | `qubit.py` | `bloch_coordinates()`, `bloch_angles()` |

---

## 🚀 Next Step: Bloch Sphere Visualizer

Now that you understand the theory, we'll build an interactive Bloch sphere visualizer where you can:
- See qubits as points on a sphere
- Apply gates and watch them rotate
- Understand geometrically what each gate does

**Ready to continue?** Let me know and I'll provide the visualization code!

---

## 💡 Pro Tips for Learning

1. **Read notes → Write code → Test it → Understand why**
2. **When stuck:** Go back to the equation in notes and trace through step-by-step
3. **Use print statements:** See what the state is at each step
4. **Draw pictures:** Sketch the Bloch sphere and trace the path
5. **Ask "why":** Why must gates be unitary? Why θ/2 in rotations?

The theory in your notes is **not abstract** - it's exactly what the code implements!