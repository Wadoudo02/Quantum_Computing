#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 Demo: Quantum Algorithms

Demonstrates all three quantum algorithms:
1. Deutsch-Jozsa - exponential speedup
2. Grover's Search - quadratic speedup
3. Quantum Fourier Transform

Author: Wadoud Charbak
For: Quantinuum & Riverlane recruitment
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from phase3_algorithms.deutsch_jozsa import deutsch_jozsa_algorithm, create_dj_circuit
from phase3_algorithms.grover import grover_search, measure_grover, optimal_grover_iterations
from phase3_algorithms.qft import quantum_fourier_transform, inverse_qft
from phase3_algorithms.oracles import deutsch_jozsa_oracle, grover_oracle
from phase3_algorithms.circuit_visualization import draw_circuit
from phase3_algorithms.performance_analysis import benchmark_deutsch_jozsa, benchmark_grover, plot_complexity_comparison


def demo_deutsch_jozsa():
    """Demonstrate Deutsch-Jozsa algorithm"""
    print("=" * 70)
    print("DEUTSCH-JOZSA ALGORITHM DEMO")
    print("=" * 70)
    print("\\nProblem: Determine if f: {0,1}^n -> {0,1} is constant or balanced")
    print("Classical: Needs up to 2^(n-1) + 1 queries")
    print("Quantum: Needs exactly 1 query!\\n")
    
    # Test 1: Constant function
    print("Test 1: Constant Function f(x) = 0")
    print("-" * 70)
    oracle = deutsch_jozsa_oracle("constant_0", 3)
    result, state, history = deutsch_jozsa_algorithm(oracle, verbose=False)
    
    print(f"  Function type: Constant")
    print(f"  Algorithm says: {result.upper()}")
    print(f"  Correct: {result == 'constant'} ✓")
    print(f"  Queries used: 1")
    print(f"  Classical would need: up to {2**(3-1) + 1} queries\\n")
    
    # Test 2: Balanced function
    print("Test 2: Balanced Function f(x) = parity(x)")
    print("-" * 70)
    oracle = deutsch_jozsa_oracle("balanced_parity", 3)
    result, state, history = deutsch_jozsa_algorithm(oracle, verbose=False)
    
    print(f"  Function type: Balanced")
    print(f"  Algorithm says: {result.upper()}")
    print(f"  Correct: {result == 'balanced'} ✓")
    print(f"  Queries used: 1")
    print(f"  Classical would need: up to {2**(3-1) + 1} queries\\n")
    
    # Draw circuit
    print("Generating circuit diagram...")
    fig, ax = draw_circuit("deutsch-jozsa", 3, save_path="plots/phase3/deutsch_jozsa_circuit.png")
    plt.close(fig)
    print("  Saved to: plots/phase3/deutsch_jozsa_circuit.png\\n")


def demo_grover():
    """Demonstrate Grover's search algorithm"""
    print("=" * 70)
    print("GROVER'S SEARCH ALGORITHM DEMO")
    print("=" * 70)
    print("\\nProblem: Search for marked item in unsorted database")
    print("Classical: Needs O(N) queries on average")
    print("Quantum: Needs O(√N) queries!\\n")
    
    n_qubits = 3
    N = 2 ** n_qubits
    target = 5
    
    print(f"Searching for |{target}⟩ in database of {N} items")
    print("-" * 70)
    
    # Calculate optimal iterations
    optimal = optimal_grover_iterations(N, 1)
    print(f"  Optimal Grover iterations: {optimal}")
    print(f"  Classical expected queries: {N/2:.0f}")
    print(f"  Quantum speedup: {(N/2)/optimal:.2f}x\\n")
    
    # Run Grover
    print("Running Grover's algorithm...")
    state, history = grover_search([target], n_qubits, verbose=False)
    
    # Measure
    measurements = measure_grover(state, shots=1000)
    from collections import Counter
    counts = Counter(measurements)
    
    print(f"\\nMeasurement Results (1000 shots):")
    for i in range(min(8, N)):
        count = counts.get(i, 0)
        prob = count / 1000
        bar = "█" * int(prob * 50)
        marker = " ← TARGET" if i == target else ""
        print(f"  |{i}⟩: {bar:<50} {prob:.3f}{marker}")
    
    success_rate = counts[target] / 1000
    print(f"\\nSuccess rate: {success_rate:.1%} ✓")
    print(f"Target amplitude: {np.abs(state[target]):.4f}\\n")
    
    # Draw circuit
    print("Generating circuit diagram...")
    fig, ax = draw_circuit("grover", 3, iterations=optimal, 
                          save_path="plots/phase3/grover_circuit.png")
    plt.close(fig)
    print("  Saved to: plots/phase3/grover_circuit.png\\n")


def demo_qft():
    """Demonstrate Quantum Fourier Transform"""
    print("=" * 70)
    print("QUANTUM FOURIER TRANSFORM DEMO")
    print("=" * 70)
    print("\\nProblem: Compute discrete Fourier transform")
    print("Classical FFT: O(n * 2^n) operations")
    print("Quantum QFT: O(n^2) operations!\\n")
    
    n = 3
    N = 2 ** n
    
    print(f"QFT on {n} qubits (N = {N} dimensional space)")
    print("-" * 70)
    
    # Test 1: QFT on |0⟩
    print("\\nTest 1: QFT|0⟩ = uniform superposition")
    state_0 = np.zeros(N)
    state_0[0] = 1.0
    
    qft_state = quantum_fourier_transform(state_0, n)
    
    print(f"  Input: |0⟩")
    print(f"  Output: Equal superposition")
    print(f"  All amplitudes equal: {np.allclose(np.abs(qft_state), 1/np.sqrt(N))} ✓")
    
    # Test 2: QFT is reversible
    print("\\nTest 2: QFT is unitary (reversible)")
    recovered = inverse_qft(qft_state, n)
    print(f"  QFT(|0⟩) then QFT^†")
    print(f"  Recovered original: {np.allclose(recovered, state_0)} ✓")
    
    # Test 3: Complexity
    print(f"\\nTest 3: Complexity comparison")
    print(f"  Classical DFT: O(n * 2^n) = O({n * N}) operations")
    print(f"  Quantum QFT: O(n^2) = O({n**2}) gates")
    print(f"  Speedup factor: {(n * N) / (n**2):.1f}x\\n")
    
    # Draw circuit
    print("Generating circuit diagram...")
    fig, ax = draw_circuit("qft", 3, save_path="plots/phase3/qft_circuit.png")
    plt.close(fig)
    print("  Saved to: plots/phase3/qft_circuit.png\\n")


def demo_performance_comparison():
    """Compare all algorithms: classical vs quantum"""
    print("=" * 70)
    print("PERFORMANCE COMPARISON: CLASSICAL VS QUANTUM")
    print("=" * 70)
    print("\\nBenchmarking algorithms on different problem sizes...\\n")
    
    # Deutsch-Jozsa
    print("1. Deutsch-Jozsa Scaling")
    print("-" * 70)
    dj_results = benchmark_deutsch_jozsa([2, 3, 4, 5], trials=20)
    
    for i, n in enumerate([2, 3, 4, 5]):
        N = 2 ** n
        q_queries = dj_results['quantum_queries'][i]
        c_avg = dj_results['classical_queries_avg'][i]
        c_worst = dj_results['classical_queries_worst'][i]
        speedup = c_avg / q_queries
        
        print(f"  n={n} (N={N:>3}): Quantum={q_queries}, Classical avg={c_avg:>4.1f}, "
              f"worst={c_worst:>3}, Speedup={speedup:>4.1f}x")
    
    # Plot DJ
    print("\\n  Generating comparison plot...")
    plot_complexity_comparison("deutsch-jozsa", dj_results, 
                              save_path="plots/phase3/dj_complexity.png")
    print("  Saved to: plots/phase3/dj_complexity.png")
    
    # Grover
    print("\\n2. Grover Scaling")
    print("-" * 70)
    grover_results = benchmark_grover([2, 3, 4], trials=50)
    
    for i, n in enumerate([2, 3, 4]):
        N = 2 ** n
        q_queries = grover_results['quantum_queries'][i]
        c_avg = N / 2
        speedup = c_avg / q_queries
        q_success = grover_results['quantum_success'][i]
        
        print(f"  n={n} (N={N:>2}): Quantum={q_queries} ({q_success:.1%}), "
              f"Classical avg={c_avg:>4.0f}, Speedup={speedup:>4.2f}x")
    
    # Plot Grover
    print("\\n  Generating comparison plot...")
    plot_complexity_comparison("grover", grover_results,
                              save_path="plots/phase3/grover_complexity.png")
    print("  Saved to: plots/phase3/grover_complexity.png\\n")


def main():
    """Run all demos"""
    print("\\n" + "=" * 70)
    print("PHASE 3: QUANTUM ALGORITHMS - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("\\nAuthor: Wadoud Charbak")
    print("Based on: Imperial College London Quantum Information Theory")
    print("For: Quantinuum & Riverlane recruitment")
    print("\\n" + "=" * 70 + "\\n")
    
    # Run all demos
    demo_deutsch_jozsa()
    print("\\n")
    
    demo_grover()
    print("\\n")
    
    demo_qft()
    print("\\n")
    
    demo_performance_comparison()
    
    print("\\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\\nGenerated files:")
    print("  - plots/phase3/deutsch_jozsa_circuit.png")
    print("  - plots/phase3/grover_circuit.png")
    print("  - plots/phase3/qft_circuit.png")
    print("  - plots/phase3/dj_complexity.png")
    print("  - plots/phase3/grover_complexity.png")
    print("\\nAll three algorithms demonstrating quantum advantage! 🚀\\n")


if __name__ == "__main__":
    main()

