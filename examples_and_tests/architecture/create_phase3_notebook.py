"""
Create Phase 3 Jupyter Notebook
"""

import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split('\n')
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split('\n')
    }

# Title
notebook["cells"].append(md_cell("""# Phase 3: Quantum Algorithms

**Demonstrating Quantum Advantage Through Three Landmark Algorithms**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Deutsch-Jozsa Algorithm](#deutsch-jozsa)
4. [Grover's Search Algorithm](#grover)
5. [Quantum Fourier Transform](#qft)
6. [Performance Comparison](#performance)
7. [Summary & Key Takeaways](#summary)

---

**Author:** Wadoud Charbak  
**Based on:** Imperial College London Quantum Information Theory  
**For:** Quantinuum & Riverlane recruitment"""))

# Introduction
notebook["cells"].append(md_cell("""---

## 1. Introduction <a id="introduction"></a>

This notebook explores three foundational quantum algorithms that demonstrate different types of quantum advantage:

### 1. **Deutsch-Jozsa Algorithm** - Exponential Speedup
- Determines if a function is constant or balanced in **1 query**
- Classical worst case: 2^(n-1) + 1 queries
- **Use case:** First algorithm showing exponential quantum speedup

### 2. **Grover's Search Algorithm** - Quadratic Speedup
- Searches unsorted database in O(√N) steps
- Classical average: O(N) steps
- **Use case:** Optimal algorithm for unstructured search

### 3. **Quantum Fourier Transform** - Exponential Speedup
- Computes Fourier transform in O(n²) gates
- Classical: O(n·2^n) operations
- **Use case:** Foundation for Shor's factoring algorithm

### What We'll Learn

- How quantum interference creates speedup
- Amplitude amplification techniques
- Oracle-based quantum computation
- Classical vs quantum complexity
- When quantum advantage exists"""))

# Setup
notebook["cells"].append(md_cell("""---

## 2. Setup <a id="setup"></a>

Import all necessary modules:"""))

notebook["cells"].append(code_cell("""# Setup paths
import sys
from pathlib import Path
notebook_dir = Path().absolute()
project_root = notebook_dir.parent
sys.path.insert(0, str(project_root / "src"))

# Core imports
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math, Latex
import warnings
warnings.filterwarnings('ignore')

# Phase 3 modules
from phase3_algorithms.deutsch_jozsa import deutsch_jozsa_algorithm, create_dj_circuit
from phase3_algorithms.grover import grover_search, measure_grover, optimal_grover_iterations
from phase3_algorithms.qft import quantum_fourier_transform, inverse_qft, qft_matrix
from phase3_algorithms.oracles import deutsch_jozsa_oracle, grover_oracle
from phase3_algorithms.circuit_visualization import draw_circuit
from phase3_algorithms.performance_analysis import benchmark_deutsch_jozsa, benchmark_grover

# Configure plots
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print("✓ Setup complete!")
print("All modules loaded successfully")"""))

# Deutsch-Jozsa
notebook["cells"].append(md_cell("""---

## 3. Deutsch-Jozsa Algorithm <a id="deutsch-jozsa"></a>

### 3.1 The Problem

Given a function f: {0,1}^n → {0,1}, determine if it is:
- **Constant:** f(x) = 0 for all x, OR f(x) = 1 for all x
- **Balanced:** f(x) = 0 for exactly half of inputs, 1 for other half

**Classical approach:** Must query f up to 2^(n-1) + 1 times in worst case

**Quantum approach:** Queries f exactly **once**!

### 3.2 Algorithm Steps

1. Initialize n qubits to |0⟩
2. Apply Hadamard to all qubits → uniform superposition
3. Apply oracle U_f → phase kickback
4. Apply Hadamard again → interference
5. Measure: if all |0⟩ → constant, else → balanced

### 3.3 Why It Works

**Key insight:** Quantum parallelism + interference

- After first Hadamard: |ψ⟩ = (1/√2^n) Σ_x |x⟩
- Oracle evaluates f(x) for ALL x simultaneously
- Second Hadamard causes interference:
  - Constant functions: constructive interference at |0⟩
  - Balanced functions: destructive interference at |0⟩"""))

notebook["cells"].append(md_cell("""### 3.4 Implementation

Let's test on 3-qubit examples:"""))

notebook["cells"].append(code_cell("""print("DEUTSCH-JOZSA ALGORITHM DEMO")
print("="*70)

# Test 1: Constant function
print("\\nTest 1: Constant Function f(x) = 0")
print("-"*70)

oracle_const = deutsch_jozsa_oracle("constant_0", 3)
result, state, history = deutsch_jozsa_algorithm(oracle_const, verbose=True)

print(f"\\n➤ Algorithm determined: {result.upper()}")
print(f"➤ Correct: {result == 'constant'} ✓")"""))

notebook["cells"].append(code_cell("""# Test 2: Balanced function
print("\\nTest 2: Balanced Function f(x) = parity(x)")
print("-"*70)

oracle_balanced = deutsch_jozsa_oracle("balanced_parity", 3)
result, state, history = deutsch_jozsa_algorithm(oracle_balanced, verbose=True)

print(f"\\n➤ Algorithm determined: {result.upper()}")
print(f"➤ Correct: {result == 'balanced'} ✓")"""))

notebook["cells"].append(md_cell("""### 3.5 Circuit Diagram

Let's visualize the quantum circuit:"""))

notebook["cells"].append(code_cell("""# Draw Deutsch-Jozsa circuit
fig, ax = draw_circuit("deutsch-jozsa", 3)
plt.suptitle("Deutsch-Jozsa Circuit (3 qubits)", fontsize=14, fontweight='bold', y=1.02)
plt.show()

print("\\nCircuit structure:")
print("  1. H gates create superposition")
print("  2. Oracle U_f encodes function")
print("  3. H gates again for interference")
print("  4. Measurement reveals answer")"""))

# Grover
notebook["cells"].append(md_cell("""---

## 4. Grover's Search Algorithm <a id="grover"></a>

### 4.1 The Problem

Search for a marked item in an unsorted database of N items.

**Classical:** Random search requires O(N) queries on average (N/2 expected)

**Quantum:** Grover's algorithm requires O(√N) queries!

### 4.2 Algorithm Components

1. **Oracle:** Marks target state by flipping phase
   - O|ω⟩ = -|ω⟩ (target)
   - O|x⟩ = |x⟩ (non-target)

2. **Diffusion Operator:** Inverts amplitudes about average
   - D = 2|s⟩⟨s| - I
   - where |s⟩ is uniform superposition

3. **Grover Iteration:** G = D·O
   - Repeat ~π/4√N times for optimal success

### 4.3 How It Works

**Amplitude Amplification:**
- Start with equal amplitudes (1/√N for each state)
- Oracle flips phase of target
- Diffusion amplifies target amplitude
- Each iteration rotates toward target

**Geometric View:**
- State vector rotates in 2D subspace
- Rotation angle: 2θ where sin(θ) = 1/√N
- After k iterations: amplitude = sin((2k+1)θ)"""))

notebook["cells"].append(md_cell("""### 4.4 Implementation

Let's search for |5⟩ in an 8-item database:"""))

notebook["cells"].append(code_cell("""print("GROVER'S SEARCH ALGORITHM DEMO")
print("="*70)

n_qubits = 3
N = 2 ** n_qubits
target = 5

print(f"\\nSearching for |{target}⟩ in database of {N} items")
print("-"*70)

# Calculate optimal iterations
optimal = optimal_grover_iterations(N, 1)
print(f"Optimal Grover iterations: {optimal}")
print(f"Classical expected queries: {N/2:.0f}")
print(f"Quantum speedup: {(N/2)/optimal:.2f}x\\n")

# Run Grover
print("Running Grover's algorithm...")
state, history = grover_search([target], n_qubits, verbose=True)

print("\\n✓ Algorithm complete!")"""))

notebook["cells"].append(code_cell("""# Measure and analyze results
measurements = measure_grover(state, shots=1000)

from collections import Counter
counts = Counter(measurements)

print("\\nMeasurement Results (1000 shots):")
print("="*70)
for i in range(N):
    count = counts.get(i, 0)
    prob = count / 1000
    bar = "█" * int(prob * 50)
    marker = " ← TARGET" if i == target else ""
    print(f"  |{i}⟩: {bar:<50} {prob:.3f}{marker}")

success_rate = counts[target] / 1000
print(f"\\nSuccess rate: {success_rate:.1%} ✓")
print(f"Target amplitude: {np.abs(state[target]):.4f}")
print(f"Theoretical max: ~0.972")"""))

notebook["cells"].append(md_cell("""### 4.5 Amplitude Evolution

Let's visualize how amplitudes change with each iteration:"""))

notebook["cells"].append(code_cell("""# Plot amplitude evolution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Left: All amplitudes over iterations
for i in range(N):
    amps = [np.abs(h[i]) for h in history]
    label = f"|{i}⟩" if i == target else None
    linewidth = 3 if i == target else 1
    color = 'red' if i == target else 'gray'
    alpha = 1.0 if i == target else 0.3
    ax1.plot(range(len(history)), amps, linewidth=linewidth, 
            color=color, alpha=alpha, label=label)

ax1.set_xlabel('Grover Iteration', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_title('Amplitude Evolution (All States)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Right: Target vs non-target
target_amps = [np.abs(h[target]) for h in history]
avg_others = [np.mean([np.abs(h[i]) for i in range(N) if i != target]) for h in history]

ax2.plot(range(len(history)), target_amps, 'r-o', linewidth=3, 
        markersize=8, label='Target |5⟩')
ax2.plot(range(len(history)), avg_others, 'b--s', linewidth=2,
        markersize=6, label='Avg others')

ax2.set_xlabel('Grover Iteration', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_title('Target vs Non-Target Amplitudes', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\\nKey observation:")
print("  • Target amplitude increases with each iteration")
print("  • Non-target amplitudes decrease")
print("  • This is amplitude amplification!")"""))

# QFT
notebook["cells"].append(md_cell("""---

## 5. Quantum Fourier Transform <a id="qft"></a>

### 5.1 The Problem

Compute the discrete Fourier transform of a quantum state.

**Classical DFT:** O(n·2^n) operations (even with FFT)

**Quantum QFT:** O(n²) quantum gates!

### 5.2 Mathematical Definition

For a quantum state |j⟩:

$$
\\text{QFT}|j\\rangle = \\frac{1}{\\sqrt{N}} \\sum_{k=0}^{N-1} e^{2\\pi ijk/N} |k\\rangle
$$

where N = 2^n.

### 5.3 Circuit Structure

For each qubit j:
1. Apply Hadamard H_j
2. Apply controlled phase rotations R_k from qubits j+1, ..., n
   - R_k = diag(1, e^(2πi/2^k))
3. SWAP qubits at end to reverse order

### 5.4 Why It's Important

- **Foundation for Shor's algorithm** (integer factorization)
- **Phase estimation** (eigenvalue problems)
- **Quantum signal processing**
- **Quantum simulation** algorithms"""))

notebook["cells"].append(md_cell("""### 5.5 Implementation

Let's test QFT properties:"""))

notebook["cells"].append(code_cell("""print("QUANTUM FOURIER TRANSFORM DEMO")
print("="*70)

n = 3
N = 2 ** n

print(f"\\nQFT on {n} qubits (N = {N} dimensional space)")
print("-"*70)

# Test 1: QFT on |0⟩
print("\\nTest 1: QFT|0⟩ = uniform superposition")
state_0 = np.zeros(N)
state_0[0] = 1.0

qft_state = quantum_fourier_transform(state_0, n)

print(f"  Input: |0⟩")
print(f"  Output amplitudes: {np.abs(qft_state[:4])}...")
print(f"  All equal: {np.allclose(np.abs(qft_state), 1/np.sqrt(N))} ✓")
print(f"  Expected: 1/√{N} = {1/np.sqrt(N):.4f}")"""))

notebook["cells"].append(code_cell("""# Test 2: QFT is unitary (reversible)
print("\\nTest 2: QFT is unitary")
print("-"*70)

recovered = inverse_qft(qft_state, n)

print(f"  Apply QFT then QFT^†")
print(f"  Recovered original: {np.allclose(recovered, state_0)} ✓")
print(f"  Error: {np.linalg.norm(recovered - state_0):.10f}")"""))

notebook["cells"].append(code_cell("""# Test 3: Visualize QFT matrix
print("\\nTest 3: QFT Matrix Structure")
print("-"*70)

QFT_matrix = qft_matrix(n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Magnitude
im1 = ax1.imshow(np.abs(QFT_matrix), cmap='viridis', aspect='auto')
ax1.set_title('QFT Matrix (Magnitude)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Input State k', fontsize=11)
ax1.set_ylabel('Output State j', fontsize=11)
plt.colorbar(im1, ax=ax1, label='|QFT[j,k]|')

# Phase
im2 = ax2.imshow(np.angle(QFT_matrix), cmap='twilight', aspect='auto')
ax2.set_title('QFT Matrix (Phase)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Input State k', fontsize=11)
ax2.set_ylabel('Output State j', fontsize=11)
plt.colorbar(im2, ax=ax2, label='arg(QFT[j,k])')

plt.tight_layout()
plt.show()

print("\\n✓ QFT matrix is unitary:")
print(f"  Unitarity check: {np.allclose(QFT_matrix @ QFT_matrix.conj().T, np.eye(N))}")"""))

# Performance Comparison
notebook["cells"].append(md_cell("""---

## 6. Performance Comparison <a id="performance"></a>

Let's benchmark all algorithms against their classical counterparts to demonstrate quantum advantage."""))

notebook["cells"].append(code_cell("""print("CLASSICAL VS QUANTUM PERFORMANCE")
print("="*70)

# Deutsch-Jozsa
print("\\n1. Deutsch-Jozsa Scaling")
print("-"*70)

dj_results = benchmark_deutsch_jozsa([2, 3, 4, 5], trials=20)

print(f"{'n':>3} | {'N':>4} | {'Quantum':>8} | {'Classical (avg)':>15} | {'Speedup':>8}")
print("-"*70)
for i, n in enumerate([2, 3, 4, 5]):
    N = 2 ** n
    q = dj_results['quantum_queries'][i]
    c = dj_results['classical_queries_avg'][i]
    speedup = c / q
    print(f"{n:>3} | {N:>4} | {q:>8} | {c:>15.1f} | {speedup:>7.1f}x")

print("\\n✓ Exponential speedup demonstrated!")"""))

notebook["cells"].append(code_cell("""# Grover
print("\\n2. Grover Scaling")
print("-"*70)

grover_results = benchmark_grover([2, 3, 4], trials=50)

print(f"{'n':>3} | {'N':>4} | {'Quantum':>8} | {'Classical':>10} | {'Speedup':>8} | {'Success':>8}")
print("-"*70)
for i, n in enumerate([2, 3, 4]):
    N = 2 ** n
    q = grover_results['quantum_queries'][i]
    c = N / 2
    speedup = c / q
    success = grover_results['quantum_success'][i]
    print(f"{n:>3} | {N:>4} | {q:>8} | {c:>10.0f} | {speedup:>7.2f}x | {success:>7.1%}")

print("\\n✓ Quadratic speedup demonstrated!")"""))

# Summary
notebook["cells"].append(md_cell("""---

## 7. Summary & Key Takeaways <a id="summary"></a>

### 7.1 What We Learned

**Three Quantum Algorithms:**

1. **Deutsch-Jozsa** - Exponential speedup (1 vs 2^(n-1)+1 queries)
2. **Grover's Search** - Quadratic speedup (√N vs N queries)
3. **Quantum Fourier Transform** - Exponential speedup (n² vs n·2^n operations)

**Key Quantum Concepts:**

- **Quantum Parallelism:** Evaluate function on all inputs simultaneously
- **Interference:** Constructive/destructive interference creates speedup
- **Amplitude Amplification:** Iteratively increase probability of correct answer
- **Phase Kickback:** Oracle encodes information as phase

### 7.2 When Quantum Advantage Exists

**Quantum algorithms excel when:**
- Problem has global structure (not local search)
- Can exploit quantum interference
- Oracle queries dominate complexity
- Need to find patterns in large spaces

**Quantum algorithms don't help when:**
- Problem requires reading all data anyway
- No quantum interference advantage
- Classical algorithms already optimal

### 7.3 Complexity Summary

| Algorithm | Classical | Quantum | Type |
|-----------|-----------|---------|------|
| Deutsch-Jozsa | O(2^n) | O(1) | Exponential |
| Grover | O(N) | O(√N) | Quadratic |
| QFT | O(n·2^n) | O(n²) | Exponential |

### 7.4 Real-World Applications

- **Deutsch-Jozsa:** Proof of concept for quantum advantage
- **Grover:** Database search, optimization, collision finding
- **QFT:** Shor's factoring, phase estimation, quantum simulation

### 7.5 Next Steps

**Phase 4:** Noise and decoherence - why quantum computers are hard to build

**Phase 5:** Quantum error correction - protecting fragile qubits

**Phase 6:** Real hardware - running on actual quantum computers

---

## 🎉 Phase 3 Complete!

You now understand:
✅ How quantum algorithms achieve speedup  
✅ Oracle-based quantum computation  
✅ Amplitude amplification techniques  
✅ When quantum advantage exists  
✅ Classical vs quantum complexity  

**Ready for error correction and real hardware!** 🚀"""))

# Write notebook
output_path = '/Users/wadoudcharbak/Documents/GitHub/Quantum_Computing/notebooks/03_phase3_quantum_algorithms.ipynb'
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook created: {output_path}")
print(f"Total cells: {len(notebook['cells'])}")

