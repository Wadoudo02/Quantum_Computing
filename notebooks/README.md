# Quantum Computing Notebooks

Interactive Jupyter notebooks for learning quantum computing concepts.

## 📚 Available Notebooks

### Phase 1: Introduction to Quantum Computing
**File:** `01_phase1_quantum_computing.ipynb`

**Topics Covered:**
- Qubit representation and state vectors
- Bloch sphere visualization
- Quantum gates (Pauli, Hadamard, rotations, phase)
- Measurement and the Born rule
- Two-qubit systems and tensor products
- Entanglement and Bell states
- Creating entanglement with CNOT

**Features:**
- ✅ Complete LaTeX equations
- ✅ Interactive code examples
- ✅ Beautiful Bloch sphere visualizations
- ✅ Step-by-step demonstrations
- ✅ Measurement statistics
- ✅ Gate sequence trajectories

---

## 🚀 How to Use

### 1. Launch Jupyter

```bash
cd notebooks
jupyter notebook
```

Or use JupyterLab:

```bash
jupyter lab
```

### 2. Open the Notebook

Click on `01_phase1_quantum_computing.ipynb` in the browser.

### 3. Run All Cells

- **Menu:** Cell → Run All
- **Keyboard:** Shift+Enter to run cells one by one

---

## 📋 Prerequisites

### Required Packages

```bash
pip install numpy matplotlib jupyter
```

### Project Structure

The notebooks use code from `src/phase1_qubits/`:

```
Quantum_Computing/
├── src/
│   └── phase1_qubits/
│       ├── qubit.py
│       ├── gates.py
│       ├── bloch_sphere.py
│       ├── multi_qubit.py
│       └── two_qubit_gates.py
├── notebooks/
│   ├── 01_phase1_quantum_computing.ipynb  ← This notebook
│   └── README.md
└── plots/
    └── phase1/  ← Visualizations saved here
```

---

## 📊 What You'll Learn

### Section 1: Quantum States
- Qubit representation as complex vectors
- Computational basis: |0⟩ and |1⟩
- Hadamard basis: |+⟩ and |-⟩
- Custom superposition states
- Normalization condition

### Section 2: Bloch Sphere
- Geometric representation of qubits
- Bloch coordinates calculation
- Visualizing basis states
- Understanding θ and φ parameters

### Section 3: Quantum Gates
- Pauli gates (X, Y, Z) as rotations
- Hadamard gate creating superposition
- Phase gates (S, T)
- Continuous rotations (Rx, Ry, Rz)
- Gate sequences and trajectories

### Section 4: Measurement
- Born rule and probabilities
- Measuring deterministic states
- Measuring superposition states
- Measurement statistics
- State collapse

### Section 5: Two-Qubit Systems
- Tensor product construction
- Separable vs entangled states
- Bell states (all 4)
- Creating entanglement with CNOT
- Schmidt decomposition
- Entanglement entropy

---

## 🎨 Visualizations

The notebook includes many interactive visualizations:

1. **Computational Basis on Bloch Sphere**
   - Blue |0⟩ at north pole
   - Red |1⟩ at south pole

2. **Hadamard Basis on Bloch Sphere**
   - Green |+⟩ on +X axis
   - Purple |-⟩ on -X axis

3. **Pauli Gate Rotations**
   - 180° rotations around X, Y, Z axes

4. **Hadamard Transformations**
   - |0⟩ → |+⟩ and |1⟩ → |-⟩

5. **Rotation Gates**
   - Rx, Ry, Rz at various angles

6. **Gate Sequence Trajectories**
   - Path of qubit under multiple gates

7. **Measurement Statistics**
   - Histograms comparing theory vs experiment

---

## 💡 Tips for Learning

### For Beginners

1. **Read the LaTeX equations** - They're rendered beautifully
2. **Run cells in order** - Each builds on previous ones
3. **Modify the code** - Change angles, states, gates
4. **Look at visualizations** - Bloch sphere is key to intuition

### For Advanced Users

1. **Experiment with custom states** - Create your own
2. **Try different gate sequences** - See trajectories
3. **Compare with theory** - Check measurement statistics
4. **Extend the code** - Add your own analysis

---

## 🔬 Key Equations

### Qubit State
```
|ψ⟩ = α|0⟩ + β|1⟩
|α|² + |β|² = 1
```

### Born Rule
```
P(0) = |α|²
P(1) = |β|²
```

### Bloch Sphere
```
x = 2Re(α*β)
y = 2Im(α*β)
z = |α|² - |β|²
```

### Bell State
```
|Φ+⟩ = (|00⟩ + |11⟩)/√2
```

---

## 📈 Expected Runtime

- **Full notebook:** ~2-3 minutes
- **Each section:** ~20-30 seconds
- **Visualizations:** ~1-2 seconds each

---

## 🐛 Troubleshooting

### ImportError: No module named 'phase1_qubits'

**Solution:** The notebook automatically adds the src directory to the path. If this fails, manually run:

```python
import sys
from pathlib import Path
sys.path.insert(0, '/path/to/Quantum_Computing/src')
```

### Plots not showing

**Solution:** Make sure you have:
```python
%matplotlib inline
```

at the top of the notebook.

### Kernel keeps dying

**Solution:** You might need more memory. Try:
- Restarting the kernel
- Closing other notebooks
- Reducing `figsize` in plots

---

## 📚 Further Reading

### After This Notebook

- **Phase 2:** Bell inequalities and CHSH
- **Phase 3:** Quantum algorithms (Deutsch-Jozsa, Simon)
- **Phase 4:** Grover's search algorithm
- **Phase 5:** Shor's factoring algorithm
- **Phase 6:** Hardware experiments

### Recommended Resources

1. **Nielsen & Chuang** - *Quantum Computation and Quantum Information*
2. **Imperial College Notes** - Sections 1.1, 1.4, 2.2, 2.4
3. **Qiskit Textbook** - Online interactive quantum computing book
4. **Our Documentation** - `../PHASE1_USAGE.md`

---

## ✨ Features

### Interactive Elements

- ✅ Live code execution
- ✅ Modifiable parameters
- ✅ Real-time visualization
- ✅ Statistical analysis

### Educational Design

- ✅ Theory before practice
- ✅ Equations with explanations
- ✅ Visual demonstrations
- ✅ Progressive complexity

### Production Quality

- ✅ Professional formatting
- ✅ Complete documentation
- ✅ Error handling
- ✅ Clean code examples

---

## 🎯 Learning Outcomes

After completing this notebook, you will be able to:

1. ✅ Represent qubits as state vectors
2. ✅ Visualize states on the Bloch sphere
3. ✅ Apply quantum gates to qubits
4. ✅ Understand measurement and probabilities
5. ✅ Work with two-qubit systems
6. ✅ Create and verify entanglement
7. ✅ Implement basic quantum circuits

---

## 🔄 Updates and Versions

**Current Version:** 1.0

**Last Updated:** October 2025

**Changelog:**
- v1.0: Initial release with complete Phase 1 content

---

## 💬 Feedback

Found an issue? Have suggestions? Please open an issue on GitHub!

---

**Happy Learning! 🚀**

*Built for Quantinuum & Riverlane Recruitment*
