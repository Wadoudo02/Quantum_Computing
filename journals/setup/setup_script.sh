#!/bin/bash
# Quantum Foundations Project Setup Script
# This script creates all necessary directories and initial files

echo "🚀 Setting up Quantum Foundations project..."
echo ""

# Create main directory structure
echo "📁 Creating directory structure..."
mkdir -p src/phase1_single_qubits
mkdir -p src/phase2_entanglement
mkdir -p src/phase3_algorithms
mkdir -p src/phase4_noise
mkdir -p src/phase5_error_correction
mkdir -p src/phase6_hardware
mkdir -p src/utils

mkdir -p notebooks
mkdir -p tests
mkdir -p examples
mkdir -p assets/demo_gifs
mkdir -p assets/screenshots
mkdir -p assets/diagrams
mkdir -p blog_posts

mkdir -p docs/theory
mkdir -p docs/tutorials
mkdir -p docs/images

# Create __init__.py files for Python packages
echo "📝 Creating Python package files..."
touch src/__init__.py
touch src/phase1_single_qubits/__init__.py
touch src/phase2_entanglement/__init__.py
touch src/phase3_algorithms/__init__.py
touch src/phase4_noise/__init__.py
touch src/phase5_error_correction/__init__.py
touch src/phase6_hardware/__init__.py
touch src/utils/__init__.py

# Create .gitignore
echo "🔒 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints

# Virtual environments
venv/
env/
ENV/
.venv

# Conda
.conda/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# OS
.DS_Store
Thumbs.db

# Credentials
.env
*.key
credentials.json

# IBM Quantum
qiskit-ibm-runtime.json
ibm-quantum-token.txt

# Large files
*.h5
*.hdf5
*.pkl
*.pickle

# Temporary files
tmp/
temp/
*.tmp
EOF

# Create LICENSE (MIT)
echo "📄 Creating LICENSE..."
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Create LEARNING_JOURNEY.md
echo "📖 Creating LEARNING_JOURNEY.md..."
cat > LEARNING_JOURNEY.md << 'EOF'
# My Quantum Computing Learning Journey

## Overview
This document tracks my learning progress, insights, challenges, and key takeaways throughout the quantum computing project.

---

## Phase 1: Single Qubits
**Start Date:** [Date]  
**Status:** Not started

### What I'm Learning
- [ ] Quantum states as vectors in Hilbert space
- [ ] Bloch sphere representation
- [ ] Single-qubit gates (Pauli X, Y, Z, Hadamard)
- [ ] Measurement and Born rule
- [ ] Time evolution

### Key Insights
*[Add your insights here as you learn]*

### Challenges Faced
*[Document any difficulties]*

### Questions Resolved
*[Keep track of what you figured out]*

---

## Phase 2: Entanglement
**Start Date:** [Date]  
**Status:** Not started

### What I'm Learning
*[To be filled in]*

---

## Phase 3: Quantum Algorithms
**Start Date:** [Date]  
**Status:** Not started

---

## Phase 4: Noise & Decoherence
**Start Date:** [Date]  
**Status:** Not started

---

## Phase 5: Error Correction
**Start Date:** [Date]  
**Status:** Not started

---

## Phase 6: Real Hardware
**Start Date:** [Date]  
**Status:** Not started

---

## Overall Reflections

### Most Surprising Discovery
*[To be filled in at the end]*

### Biggest Challenge Overcome
*[To be filled in at the end]*

### What I'd Do Differently
*[To be filled in at the end]*

### Next Steps in Quantum Computing
*[To be filled in at the end]*
EOF

# Create setup.py for package installation
echo "⚙️  Creating setup.py..."
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="quantum-foundations",
    version="0.1.0",
    description="Interactive toolkit for learning quantum computing fundamentals",
    author="Wadoud Charbak",
    author_email="wcharbak@icloud.com",
    packages=find_packages(),
    install_requires=[
        "qiskit>=1.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "jupyter>=1.0.0",
    ],
    python_requires=">=3.9",
)
EOF

# Create initial test file
echo "🧪 Creating initial test structure..."
cat > tests/test_phase1.py << 'EOF'
"""
Unit tests for Phase 1: Single Qubits
"""
import pytest
import numpy as np

def test_placeholder():
    """Placeholder test to verify pytest is working"""
    assert True

# More tests will be added as we implement Phase 1
EOF

echo ""
echo "✅ Project structure created successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Create conda environment: conda env create -f environment.yml"
echo "2. Activate environment: conda activate quantum-computing"
echo "3. Verify installation: python -c 'import qiskit; print(qiskit.__version__)'"
echo "4. Start coding in src/phase1_single_qubits/"
echo ""
echo "Happy quantum computing! 🎉"
