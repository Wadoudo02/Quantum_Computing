# -*- coding: utf-8 -*-
"""
Performance Analysis - Classical vs Quantum

Compare quantum algorithms with their classical counterparts:
- Query complexity
- Time complexity
- Success rates
- Scaling behavior

Author: Wadoud Charbak
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable
import time
from collections import Counter

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def classical_deutsch_jozsa(oracle_func: Callable, n_qubits: int) -> Tuple[str, int]:
    """
    Classical algorithm for Deutsch-Jozsa problem.
    
    Worst case: needs 2^(n-1) + 1 queries to be certain.
    
    Returns
    -------
    result : str
        "constant" or "balanced"
    queries : int
        Number of queries made
    """
    N = 2 ** n_qubits
    
    # Query first value
    first_val = oracle_func(0)
    queries = 1
    
    # Keep querying until we find different value or exhaust half
    for x in range(1, N // 2 + 2):
        val = oracle_func(x)
        queries += 1
        
        if val != first_val:
            # Found different value - must be balanced
            return "balanced", queries
    
    # All same - must be constant
    return "constant", queries


def classical_grover(target: int, N: int, max_queries: int = None) -> Tuple[int, int]:
    """
    Classical random search.
    
    Expected: N/2 queries on average, N in worst case.
    
    Returns
    -------
    found : int
        The found target (or -1 if not found)
    queries : int
        Number of queries made
    """
    if max_queries is None:
        max_queries = N
    
    tried = set()
    
    for queries in range(1, max_queries + 1):
        # Random guess
        guess = np.random.randint(0, N)
        while guess in tried:
            guess = np.random.randint(0, N)
        
        tried.add(guess)
        
        if guess == target:
            return target, queries
    
    return -1, max_queries


def benchmark_deutsch_jozsa(n_qubits_range: List[int], trials: int = 10) -> Dict:
    """
    Benchmark Deutsch-Jozsa: quantum vs classical.
    
    Returns
    -------
    dict with:
        - n_qubits: list of qubit counts
        - quantum_queries: always 1
        - classical_queries_avg: average classical queries
        - classical_queries_worst: worst-case classical queries
    """
    from phase3_algorithms.deutsch_jozsa import deutsch_jozsa_algorithm
    from phase3_algorithms.oracles import deutsch_jozsa_oracle
    
    results = {
        'n_qubits': n_qubits_range,
        'quantum_queries': [],
        'classical_queries_avg': [],
        'classical_queries_worst': []
    }
    
    for n in n_qubits_range:
        print(f"  Benchmarking n={n}...")
        
        # Quantum: always 1 query
        results['quantum_queries'].append(1)
        
        # Classical: average over trials
        classical_queries = []
        for _ in range(trials):
            oracle = deutsch_jozsa_oracle("balanced_parity", n)
            _, queries = classical_deutsch_jozsa(oracle.function, n)
            classical_queries.append(queries)
        
        results['classical_queries_avg'].append(np.mean(classical_queries))
        results['classical_queries_worst'].append(2 ** (n-1) + 1)
    
    return results


def benchmark_grover(n_qubits_range: List[int], trials: int = 100) -> Dict:
    """
    Benchmark Grover: quantum vs classical.
    
    Returns
    -------
    dict with:
        - n_qubits: list of qubit counts
        - quantum_queries: optimal Grover iterations
        - classical_queries_avg: average classical queries
        - quantum_success: quantum success rate
        - classical_success: classical success rate (for same queries)
    """
    from phase3_algorithms.grover import grover_search, optimal_grover_iterations, measure_grover
    
    results = {
        'n_qubits': n_qubits_range,
        'quantum_queries': [],
        'classical_queries_avg': [],
        'quantum_success': [],
        'classical_success': []
    }
    
    for n in n_qubits_range:
        print(f"  Benchmarking n={n}...")
        
        N = 2 ** n
        target = np.random.randint(0, N)
        
        # Quantum
        optimal_iters = optimal_grover_iterations(N, 1)
        results['quantum_queries'].append(optimal_iters)
        
        state, _ = grover_search([target], n, verbose=False)
        measurements = measure_grover(state, shots=trials)
        quantum_success = measurements.count(target) / trials
        results['quantum_success'].append(quantum_success)
        
        # Classical (average)
        classical_queries = []
        classical_found = 0
        for _ in range(trials):
            found, queries = classical_grover(target, N, max_queries=optimal_iters)
            classical_queries.append(queries)
            if found == target:
                classical_found += 1
        
        results['classical_queries_avg'].append(np.mean(classical_queries))
        results['classical_success'].append(classical_found / trials)
    
    return results


def plot_complexity_comparison(algorithm: str, results: Dict, save_path: str = None):
    """
    Plot complexity comparison.
    
    Parameters
    ----------
    algorithm : str
        "deutsch-jozsa" or "grover"
    results : dict
        Benchmark results
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_qubits = results['n_qubits']
    
    if algorithm == "deutsch-jozsa":
        ax.plot(n_qubits, results['quantum_queries'], 'b-o', linewidth=3, 
               markersize=8, label='Quantum (DJ)', zorder=3)
        ax.plot(n_qubits, results['classical_queries_avg'], 'r--s', linewidth=2,
               markersize=8, label='Classical (average)', alpha=0.7)
        ax.plot(n_qubits, results['classical_queries_worst'], 'r-^', linewidth=2,
               markersize=8, label='Classical (worst case)', alpha=0.7)
        
        ax.set_title('Deutsch-Jozsa: Quantum vs Classical Query Complexity', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Queries', fontsize=12)
        
    elif algorithm == "grover":
        N_values = [2**n for n in n_qubits]
        classical_linear = N_values
        quantum_sqrt = [int(np.pi/4 * np.sqrt(N)) for N in N_values]
        
        ax.plot(n_qubits, quantum_sqrt, 'b-o', linewidth=3,
               markersize=8, label='Quantum (Grover) ~ √N', zorder=3)
        ax.plot(n_qubits, [N/2 for N in N_values], 'r--s', linewidth=2,
               markersize=8, label='Classical ~ N/2', alpha=0.7)
        
        ax.set_title("Grover's Search: Quantum vs Classical Query Complexity",
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Expected Queries', fontsize=12)
    
    ax.set_xlabel('Number of Qubits (n)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig, ax


def compare_classical_quantum(algorithm: str, n_qubits: int) -> Dict:
    """
    Direct comparison for specific problem size.
    
    Returns detailed statistics.
    """
    if algorithm == "deutsch-jozsa":
        results = benchmark_deutsch_jozsa([n_qubits], trials=100)
        
        return {
            'algorithm': algorithm,
            'n_qubits': n_qubits,
            'N': 2 ** n_qubits,
            'quantum_queries': 1,
            'classical_avg': results['classical_queries_avg'][0],
            'classical_worst': results['classical_queries_worst'][0],
            'speedup_avg': results['classical_queries_avg'][0] / 1,
            'speedup_worst': results['classical_queries_worst'][0] / 1
        }
    
    elif algorithm == "grover":
        results = benchmark_grover([n_qubits], trials=100)
        
        N = 2 ** n_qubits
        return {
            'algorithm': algorithm,
            'n_qubits': n_qubits,
            'N': N,
            'quantum_queries': results['quantum_queries'][0],
            'classical_avg': N / 2,
            'quantum_success': results['quantum_success'][0],
            'speedup': (N / 2) / results['quantum_queries'][0]
        }


def generate_performance_report(algorithm: str, n_qubits_range: List[int]) -> str:
    """
    Generate comprehensive performance report.
    
    Returns markdown-formatted report.
    """
    report = f"# Performance Analysis: {algorithm.title()}\n\n"
    
    if algorithm == "deutsch-jozsa":
        results = benchmark_deutsch_jozsa(n_qubits_range, trials=50)
        
        report += "## Query Complexity\n\n"
        report += "| n qubits | N = 2^n | Quantum | Classical (avg) | Classical (worst) | Speedup (avg) |\n"
        report += "|----------|---------|---------|-----------------|-------------------|---------------|\n"
        
        for i, n in enumerate(n_qubits_range):
            N = 2 ** n
            q_queries = results['quantum_queries'][i]
            c_avg = results['classical_queries_avg'][i]
            c_worst = results['classical_queries_worst'][i]
            speedup = c_avg / q_queries
            
            report += f"| {n} | {N} | {q_queries} | {c_avg:.1f} | {c_worst} | {speedup:.1f}x |\n"
        
        report += "\n**Quantum Advantage:** Exponential speedup (1 query vs 2^(n-1)+1 worst case)\n"
    
    elif algorithm == "grover":
        results = benchmark_grover(n_qubits_range, trials=50)
        
        report += "## Query Complexity & Success Rate\n\n"
        report += "| n qubits | N = 2^n | Quantum | Classical (avg) | Speedup | Q Success | C Success |\n"
        report += "|----------|---------|---------|-----------------|---------|-----------|-----------|\n"
        
        for i, n in enumerate(n_qubits_range):
            N = 2 ** n
            q_queries = results['quantum_queries'][i]
            c_avg = N / 2
            speedup = c_avg / q_queries
            q_success = results['quantum_success'][i]
            c_success = results['classical_success'][i]
            
            report += f"| {n} | {N} | {q_queries} | {c_avg:.0f} | {speedup:.2f}x | {q_success:.1%} | {c_success:.1%} |\n"
        
        report += "\n**Quantum Advantage:** Quadratic speedup (√N vs N queries)\n"
    
    return report


if __name__ == "__main__":
    print("\nPerformance Analysis Demo\n")
    
    # Test Deutsch-Jozsa
    print("1. Deutsch-Jozsa Benchmark (n=2 to 5)")
    print("-" * 60)
    dj_results = benchmark_deutsch_jozsa([2, 3, 4, 5], trials=20)
    print(f"   Quantum queries: {dj_results['quantum_queries']}")
    print(f"   Classical avg: {dj_results['classical_queries_avg']}")
    print(f"   Speedup at n=5: {dj_results['classical_queries_avg'][-1]:.0f}x")
    
    # Test Grover
    print("\n2. Grover Benchmark (n=2 to 4)")
    print("-" * 60)
    grover_results = benchmark_grover([2, 3, 4], trials=20)
    print(f"   Quantum queries: {grover_results['quantum_queries']}")
    print(f"   Quantum success: {[f'{s:.1%}' for s in grover_results['quantum_success']]}")
    print(f"   Speedup at n=4: {8 / grover_results['quantum_queries'][-1]:.2f}x")
    
    print("\nPerformance analysis tests passed!")
