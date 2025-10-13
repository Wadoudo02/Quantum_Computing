# Quick Start Guide

**Get started with Phase 1 in 60 seconds!**

## Option 1: Interactive Web App (Recommended)

```bash
cd src/phase1_qubits
streamlit run app.py
```

🎯 **Best for**: Interactive exploration, demos, visual learning

---

## Option 2: Complete Demo

```bash
python examples/phase1_complete_demo.py
```

🎯 **Best for**: Comprehensive overview, understanding all features

---

## Option 3: Quick Code Examples

### Create and Visualize a Qubit

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from phase1_qubits.qubit import ket_0
from phase1_qubits.gates import HADAMARD, apply_gate
from phase1_qubits.bloch_sphere import BlochSphere

# Create qubit and apply Hadamard
q = ket_0()
q_super = apply_gate(q, HADAMARD)

# Visualize
bloch = BlochSphere()
bloch.add_qubit(q, label="|0⟩", color='blue')
bloch.add_qubit(q_super, label="|+⟩", color='red')
bloch.show()
```

### Create Bell State

```python
from phase1_qubits.multi_qubit import bell_phi_plus

bell = bell_phi_plus()
print(f"Is entangled? {bell.is_entangled()}")
print(f"Entropy: {bell.entanglement_entropy():.4f}")
```

---

## Running Tests

```bash
# Test Bloch sphere
python examples/test_bloch_sphere.py

# Test Streamlit app
python examples/test_streamlit_app.py
```

---

## File Reference

| File | Purpose |
|------|---------|
| `src/phase1_qubits/qubit.py` | Qubit class |
| `src/phase1_qubits/gates.py` | Quantum gates |
| `src/phase1_qubits/bloch_sphere.py` | Visualization |
| `src/phase1_qubits/app.py` | Web app |
| `journals/phase1/PHASE1_USAGE.md` | Full usage guide |
| `journals/phase1/PHASE1_COMPLETE_SUMMARY.md` | What's been built |

---

## Need Help?

- **Usage Guide**: See [journals/phase1/PHASE1_USAGE.md](journals/phase1/PHASE1_USAGE.md)
- **Architecture**: See [docs/architecture.md](docs/architecture.md)
- **Complete Summary**: See [journals/phase1/PHASE1_COMPLETE_SUMMARY.md](journals/phase1/PHASE1_COMPLETE_SUMMARY.md)

---

**Start with the Streamlit app - it's the easiest way to see everything!**
