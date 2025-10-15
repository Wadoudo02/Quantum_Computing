# Examples and Tests

This directory contains all example scripts, demonstrations, and test files organized by phase.

## 📁 Directory Structure

```
examples_and_tests/
├── phase1/          # Phase 1 examples and tests
├── phase2/          # Phase 2 examples and tests
├── phase3/          # Phase 3 examples and tests
├── architecture/    # Build scripts and architecture demos
└── README.md        # This file
```

---

## 🚀 Running Examples

All scripts should be run from the **project root directory**:

```bash
# Phase 1 Examples
python examples_and_tests/phase1/phase1_quick_demo.py
python examples_and_tests/phase1/phase1_complete_demo.py
python examples_and_tests/phase1/test_phase1.py

# Phase 2 Examples
python examples_and_tests/phase2/phase2_demo.py

# Phase 3 Examples
python examples_and_tests/phase3/phase3_demo.py
python examples_and_tests/phase3/test_phase3_integration.py
```

---

## 📂 Phase 1 - Single Qubits & Basic Gates

| File | Description | Run Time |
|------|-------------|----------|
| **phase1_quick_demo.py** | Fast non-interactive demo | ~2 sec |
| **phase1_complete_demo.py** | Full demonstration with plots | ~10 sec |
| **test_phase1.py** | Unit tests for Phase 1 | ~1 sec |
| **test_bloch_sphere.py** | Bloch sphere visualization tests | ~5 sec |
| **test_streamlit_app.py** | Streamlit app component tests | ~3 sec |
| **two_qubit_demo.py** | Two-qubit operations demo | ~2 sec |

### Key Features Demonstrated
- Single qubit states (|0⟩, |1⟩, |+⟩, |−⟩)
- Quantum gates (Pauli, Hadamard, rotation)
- Bloch sphere visualization
- Measurement and Born rule
- Two-qubit systems and CNOT
- Bell state creation

---

## 📂 Phase 2 - Entanglement & Bell's Inequality

| File | Description | Run Time |
|------|-------------|----------|
| **phase2_demo.py** | Bell's inequality violation demo | ~30 sec |

### Key Features Demonstrated
- All four Bell states (|Φ±⟩, |Ψ±⟩)
- Entanglement measures (Schmidt decomposition, von Neumann entropy)
- CHSH inequality test
- Classical vs quantum correlations
- Experimental violation demonstration

---

## 📂 Phase 3 - Quantum Algorithms

| File | Description | Run Time |
|------|-------------|----------|
| **phase3_demo.py** | All three algorithms with benchmarks | ~60 sec |
| **test_phase3_integration.py** | Integration test with Phases 1 & 2 | ~5 sec |

### Key Features Demonstrated
- **Deutsch-Jozsa:** Exponential speedup (1 vs 2^(n-1)+1 queries)
- **Grover's Search:** Quadratic speedup (√N vs N queries)
- **Quantum Fourier Transform:** Exponential speedup (n² vs n·2^n)
- Circuit visualization
- Performance analysis
- Classical vs quantum comparison

---

## 📂 Architecture - Build Scripts

| File | Description |
|------|-------------|
| **architecture_demo.py** | System architecture demonstration |
| **create_all_phase3.py** | Script that generated Phase 3 modules |
| **create_phase3_modules.py** | Module generation utilities |
| **create_phase3_notebook.py** | Jupyter notebook generator |
| **create_visualization_and_analysis.py** | Viz/analysis module generator |

**Note:** These are the scripts used to programmatically create Phase 3 modules. Kept for reference and reproducibility.

---

## 🧪 Testing

### Quick Tests
```bash
# Test each phase quickly
python examples_and_tests/phase1/test_phase1.py
python examples_and_tests/phase3/test_phase3_integration.py
```

### Full Integration Test
```bash
# Comprehensive test of all phases
python examples_and_tests/phase3/test_phase3_integration.py
```

Expected output:
- ✅ Phase 1 gates integrated
- ✅ Phase 2 concepts applied  
- ✅ All 3 Phase 3 algorithms functional
- ✅ Circuit visualization working
- ✅ Performance analysis operational

---

## 📊 Generated Outputs

All demos generate visualizations saved to:
```
plots/
├── phase1/   # Bloch spheres, gate trajectories
├── phase2/   # CHSH plots, Bell state matrices
└── phase3/   # Circuit diagrams, complexity plots
```

---

## 💡 Tips

1. **Always run from project root** - Scripts use relative paths
2. **Check plots/** - Visualizations are saved automatically
3. **Use quick demos first** - Faster verification
4. **Full demos for presentations** - More comprehensive output

---

## 🎓 For Recruiters

### Quick Demo Path (5 minutes)

```bash
# 1. Quick test everything works
python examples_and_tests/phase3/test_phase3_integration.py

# 2. Show Phase 3 algorithms
python examples_and_tests/phase3/phase3_demo.py

# 3. View generated plots
ls -lh plots/phase3/
```

### Full Demo Path (15 minutes)

```bash
# 1. Phase 1 complete demo
python examples_and_tests/phase1/phase1_complete_demo.py

# 2. Phase 2 Bell's inequality
python examples_and_tests/phase2/phase2_demo.py

# 3. Phase 3 quantum algorithms
python examples_and_tests/phase3/phase3_demo.py

# 4. Show all visualizations
open plots/phase*/*.png
```

---

## 📚 Documentation

For detailed information about each phase, see:
- **Phase 1:** `src/phase1_qubits/README.md`
- **Phase 2:** `src/phase2_entanglement/README.md`
- **Phase 3:** `src/phase3_algorithms/README.md`

For overall project structure:
- **Quick Start:** `QUICK_START.md`
- **Master Plan:** `quantum_master_plan.md`
- **Main README:** `README.md`

---

**All examples tested and working!** ✅
