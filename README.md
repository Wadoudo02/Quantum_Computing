# Quantum Computing Foundations: An Interactive Learning Suite

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-6133BD.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A comprehensive, interactive toolkit for learning quantum computing from first principles. Built as a hands-on learning project to understand the fundamentals of quantum information theory, quantum algorithms, and quantum error correction.

---

## 🎯 Project Overview

This project represents a systematic journey through quantum computing fundamentals, implementing key concepts from quantum mechanics, information theory, and quantum algorithms. Each module builds progressively from single-qubit operations to multi-qubit entanglement, quantum algorithms, noise modelling, and error correction.

**Target Applications:** Quantum computing research, quantum error correction, and quantum algorithm development.

**Built for:** Understanding the theoretical foundations and practical challenges facing near-term quantum computers (NISQ era).

---

## ✨ Key Features

### 🔬 **Phase 1: Single Qubits**
- Interactive Bloch sphere visualisation
- Implementation of fundamental quantum gates (Pauli X, Y, Z, Hadamard)
- Time evolution under arbitrary Hamiltonians
- Measurement simulation with Born rule probabilities
- Real-time visualisation of quantum state transformations

### 🔗 **Phase 2: Entanglement & Bell States**
- Bell state creation and verification
- Partial trace and reduced density matrices
- Schmidt decomposition implementation
- Bell's inequality demonstration (CHSH inequality)
- Entanglement quantification (von Neumann entropy)

### ⚡ **Phase 3: Quantum Algorithms**
- Deutsch-Jozsa algorithm with step-by-step visualisation
- Grover's search algorithm (2-3 qubit systems)
- Quantum Fourier Transform implementation
- Circuit diagram generation and optimisation
- Classical vs quantum comparison framework

### 🌊 **Phase 4: Noise & Decoherence**
- Density matrix formalism
- Quantum channel implementation (bit-flip, phase-flip, depolarising)
- Decoherence simulation with T₁/T₂ times
- Fidelity decay visualisation
- Impact analysis on entangled states

### 🛡️ **Phase 5: Quantum Error Correction**
- 3-qubit bit-flip code implementation
- Shor's 9-qubit code
- Stabiliser formalism and syndrome measurement
- Error detection and recovery operations
- Success rate analysis under various noise models

### 💻 **Phase 6: Real Quantum Hardware & NISQ Computing**
- Multi-platform hardware interfaces (IBM, IonQ, Rigetti)
- Realistic noise models based on actual hardware specs
- Circuit transpilation and optimization for hardware constraints
- Error mitigation techniques (ZNE, PEC, readout calibration)
- Hardware benchmarking (Randomized Benchmarking, Quantum Volume, T₁/T₂)
- NISQ algorithms (VQE, QAOA, Quantum Teleportation)
- Comprehensive analysis and visualization tools

---

## 🚀 Quick Start

### Installation

**Requirements:**
- Python 3.9 or higher
- Conda (recommended) or pip

**1. Clone the repository:**
```bash
git clone https://github.com/Wadoudo02/Quantum_Computing.git
cd Quantum_Computing
```

**2. Create conda environment:**
```bash
conda env create -f environment.yml
conda activate quantum-computing
```

**3. Run interactive demos:**
```bash
# Single qubit visualiser
streamlit run src/phase1_single_qubits/app.py

# Entanglement explorer
streamlit run src/phase2_entanglement/app.py

# Algorithm playground
streamlit run src/phase3_algorithms/app.py
```

**4. Explore Jupyter notebooks:**
```bash
jupyter notebook notebooks/
```

---

## 📚 Project Structure

```
quantum-foundations/
├── src/                          # Source code modules
│   ├── phase1_single_qubits/     # Single qubit operations
│   ├── phase2_entanglement/      # Two-qubit systems
│   ├── phase3_algorithms/        # Quantum algorithms
│   ├── phase4_noise/             # Noise and decoherence
│   ├── phase5_error_correction/  # Error correction codes
│   └── phase6_hardware/          # Real hardware integration
├── notebooks/                     # Interactive Jupyter notebooks
├── tests/                         # Unit tests
├── docs/                          # Documentation and theory
└── examples/                      # Example scripts
```

---

## 🎓 Learning Journey

This project follows a structured learning path through quantum computing:

1. **Quantum States & Operators** → Understanding qubits, gates, and the Bloch sphere
2. **Entanglement & Measurement** → Two-qubit systems and quantum correlations
3. **Quantum Algorithms** → Leveraging interference for computational advantage
4. **Decoherence & Noise** → Understanding real quantum system limitations
5. **Error Correction** → Protecting quantum information from noise
6. **Hardware Reality** → Running on actual quantum processors

Each phase includes:
- ✅ Theoretical foundations with mathematical derivations
- ✅ Python implementations with clear documentation
- ✅ Interactive visualisations
- ✅ Jupyter notebooks with explanations
- ✅ Unit tests verifying correctness

---

## 🔧 Technical Stack

**Quantum Computing:**
- [Qiskit](https://qiskit.org/) - IBM's quantum computing framework
- Qiskit Aer - High-performance simulators
- IBM Quantum - Real hardware access

**Scientific Computing:**
- NumPy - Linear algebra and numerical operations
- SciPy - Advanced scientific computing

**Visualisation:**
- Matplotlib - Static plots and diagrams
- Plotly - Interactive 3D visualisations
- Streamlit - Web-based interactive interfaces

**Development:**
- Pytest - Testing framework
- Black - Code formatting
- Jupyter - Interactive notebooks

---

## 📊 Demonstrations

### Single Qubit on Bloch Sphere
*[Demo GIF - Qubit rotation under Pauli gates]*

### Bell State Entanglement
*[Demo GIF - Creating and measuring Bell states]*

### Grover's Algorithm
*[Demo GIF - Quantum search with amplitude amplification]*

### Error Correction
*[Demo GIF - Detecting and correcting quantum errors]*

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific phase tests
pytest tests/test_phase1.py -v

# Run with coverage report
pytest --cov=src tests/
```

---

## 📖 Documentation

Detailed documentation for each phase is available in the `docs/` directory:

- [Phase 1: Single Qubits](docs/theory/01_single_qubits.md)
- [Phase 2: Entanglement](docs/theory/02_entanglement.md)
- [Phase 3: Algorithms](docs/theory/03_algorithms.md)
- [Phase 4: Noise & Decoherence](docs/theory/04_noise.md)
- [Phase 5: Error Correction](docs/theory/05_error_correction.md)
- [Phase 6: Real Hardware & NISQ](src/phase6_hardware/README.md)
- [Hardware Setup Guide](src/phase6_hardware/HARDWARE_GUIDE.md)
- [For Recruiters](src/phase6_hardware/FOR_RECRUITERS.md)

---

## 🎯 Key Concepts Implemented

### Quantum Mechanics Fundamentals
- Hilbert space formalism
- Unitary evolution
- Projective measurements
- Born rule probabilities

### Quantum Information Theory
- Entanglement measures
- Density matrices
- Quantum channels (Kraus operators)
- Fidelity and trace distance

### Quantum Algorithms
- Quantum parallelism
- Interference and amplitude amplification
- Quantum Fourier Transform
- Oracle-based algorithms

### Quantum Error Correction
- Stabiliser codes
- Syndrome measurement
- Error recovery operations
- Fault-tolerant quantum computing principles

---

## 🌟 Applications & Extensions

This toolkit provides foundations for:
- **Quantum Algorithm Research** - Implementing and testing new quantum algorithms
- **Error Correction Development** - Testing novel error correction schemes
- **Quantum Hardware Characterisation** - Benchmarking real quantum devices
- **Education & Outreach** - Teaching quantum computing concepts interactively

---

## 🤝 Contributing

This is a personal learning project, but suggestions and improvements are welcome! If you find issues or have ideas for enhancements:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request

---

## 📚 References & Resources

**Theoretical Foundations:**
- Imperial College London - Quantum Information Theory Course Materials
- Nielsen & Chuang - *Quantum Computation and Quantum Information*
- Preskill - *Quantum Computation* Lecture Notes

**Technical Documentation:**
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [IBM Quantum Experience](https://quantum-computing.ibm.com/)

**Online Resources:**
- [Quantum Country](https://quantum.country/) - Interactive quantum computing essays
- [Qiskit Textbook](https://qiskit.org/textbook/) - Open-source quantum computing education

---

## 📝 Learning Notes

Throughout this project, I documented my learning journey, challenges faced, and insights gained. Key takeaways include:

- **Superposition isn't magic** - It's a natural consequence of quantum mechanics' linear algebra structure
- **Entanglement is correlation** - But stronger than any classical correlation can be
- **Quantum algorithms are rare** - Finding quantum advantage requires specific problem structures
- **Decoherence is the enemy** - Real quantum computers face constant battle against environmental noise
- **Error correction is expensive** - Current overhead makes logical qubits incredibly costly

Detailed learning notes available in [LEARNING_JOURNEY.md](LEARNING_JOURNEY.md).

---

## 📧 Contact

**Author:** Wadoud Charbak 
**Email:** wcharbak@icloud.com
**LinkedIn:** https://www.linkedin.com/in/wadoud-charbak/


**Interested in:** Quantum computing, quantum error correction, quantum algorithms, and quantum hardware development.

**Currently seeking:** Internship opportunities in quantum computing (Summer 2025).

---

## 📄 Licence

This project is licenced under the MIT Licence - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Imperial College London** - Course materials and theoretical foundations
- **Qiskit Community** - Excellent documentation and support
- **IBM Quantum** - Access to real quantum hardware
- **Quantum Computing Community** - Online resources and discussions

---

## ✅ Project Complete!

This comprehensive quantum computing project now includes:

- ✅ **14,800+ lines** of production-quality Python code
- ✅ **Complete learning journey** from qubits to real hardware
- ✅ **6 phases** covering theory, algorithms, noise, error correction, and NISQ computing
- ✅ **NISQ algorithms** including VQE and QAOA implementations
- ✅ **Error mitigation** techniques for near-term quantum computers
- ✅ **Multi-platform support** for IBM, IonQ, and Rigetti backends

### Potential Future Extensions:

- [ ] Run algorithms on actual IBM Quantum hardware (account setup included)
- [ ] Topological error correction codes (surface codes, color codes)
- [ ] Advanced quantum machine learning demonstrations
- [ ] Quantum simulation applications (chemistry, materials)
- [ ] Real-time hardware experiment automation

---

**Built with curiosity, Python, and a lot of linear algebra** 🧮✨

*Last updated: [Current Date]*