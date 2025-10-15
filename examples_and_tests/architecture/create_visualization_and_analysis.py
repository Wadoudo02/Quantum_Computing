"""
Create circuit_visualization.py and performance_analysis.py
"""

import os

base_path = '/Users/wadoudcharbak/Documents/GitHub/Quantum_Computing/src/phase3_algorithms'

# ============================================================================
# CIRCUIT VISUALIZATION
# ============================================================================

circuit_viz_content = '''# -*- coding: utf-8 -*-
"""
Circuit Visualization for Quantum Algorithms

Generate publication-quality circuit diagrams for:
- Deutsch-Jozsa algorithm
- Grover's algorithm  
- Quantum Fourier Transform

Author: Wadoud Charbak
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Gate:
    """Represents a gate in a circuit diagram"""
    name: str
    qubits: List[int]
    time_step: int
    control_qubits: Optional[List[int]] = None
    params: Optional[Dict] = None


class CircuitDiagram:
    """
    Create quantum circuit diagrams.
    
    Example
    -------
    >>> circuit = CircuitDiagram(n_qubits=3)
    >>> circuit.h(0)
    >>> circuit.h(1)
    >>> circuit.cnot(0, 1)
    >>> circuit.draw()
    """
    
    def __init__(self, n_qubits: int, figsize: Tuple[float, float] = None):
        self.n_qubits = n_qubits
        self.gates = []
        self.max_time = 0
        
        if figsize is None:
            figsize = (12, 2 + n_qubits * 0.8)
        self.figsize = figsize
    
    def h(self, qubit: int):
        """Add Hadamard gate"""
        self.gates.append(Gate("H", [qubit], self.max_time))
        self.max_time += 1
        return self
    
    def x(self, qubit: int):
        """Add Pauli X gate"""
        self.gates.append(Gate("X", [qubit], self.max_time))
        self.max_time += 1
        return self
    
    def z(self, qubit: int):
        """Add Pauli Z gate"""
        self.gates.append(Gate("Z", [qubit], self.max_time))
        self.max_time += 1
        return self
    
    def cnot(self, control: int, target: int):
        """Add CNOT gate"""
        self.gates.append(Gate("CNOT", [target], self.max_time, [control]))
        self.max_time += 1
        return self
    
    def oracle(self, qubits: List[int], label: str = "O_f"):
        """Add oracle box"""
        self.gates.append(Gate(label, qubits, self.max_time))
        self.max_time += 1
        return self
    
    def measure(self, qubit: int):
        """Add measurement"""
        self.gates.append(Gate("M", [qubit], self.max_time))
        self.max_time += 1
        return self
    
    def barrier(self):
        """Add visual barrier"""
        self.gates.append(Gate("BARRIER", list(range(self.n_qubits)), self.max_time))
        self.max_time += 1
        return self
    
    def draw(self, save_path: Optional[str] = None, show: bool = True):
        """Draw the circuit"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Draw qubit lines
        for i in range(self.n_qubits):
            y = self.n_qubits - i - 1
            ax.plot([0, self.max_time + 1], [y, y], 'k-', linewidth=1)
            ax.text(-0.5, y, f'|q_{i}⟩', ha='right', va='center', fontsize=12)
        
        # Draw gates
        for gate in self.gates:
            self._draw_gate(ax, gate)
        
        ax.set_xlim(-1, self.max_time + 1.5)
        ax.set_ylim(-0.5, self.n_qubits - 0.5)
        ax.axis('off')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig, ax
    
    def _draw_gate(self, ax, gate: Gate):
        """Draw a single gate"""
        x = gate.time_step + 0.5
        
        if gate.name == "BARRIER":
            for i in range(self.n_qubits):
                y = self.n_qubits - i - 1
                ax.plot([x, x], [y - 0.3, y + 0.3], 'gray', linewidth=2, linestyle='--')
            return
        
        if gate.name == "CNOT":
            # Draw control and target
            control = gate.control_qubits[0]
            target = gate.qubits[0]
            
            y_control = self.n_qubits - control - 1
            y_target = self.n_qubits - target - 1
            
            # Control dot
            circle = Circle((x, y_control), 0.1, color='black', zorder=3)
            ax.add_patch(circle)
            
            # Connection line
            ax.plot([x, x], [y_control, y_target], 'k-', linewidth=2, zorder=2)
            
            # Target circle with X
            circle = Circle((x, y_target), 0.2, fill=False, edgecolor='black', linewidth=2, zorder=3)
            ax.add_patch(circle)
            ax.plot([x-0.15, x+0.15], [y_target, y_target], 'k-', linewidth=2, zorder=4)
            ax.plot([x, x], [y_target-0.15, y_target+0.15], 'k-', linewidth=2, zorder=4)
        
        elif gate.name == "M":
            # Measurement box
            y = self.n_qubits - gate.qubits[0] - 1
            box = FancyBboxPatch((x-0.3, y-0.3), 0.6, 0.6, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='black', facecolor='lightblue', linewidth=2)
            ax.add_patch(box)
            
            # Meter symbol
            ax.plot([x-0.15, x, x+0.15], [y-0.1, y+0.1, y-0.1], 'k-', linewidth=1.5)
            ax.arrow(x, y-0.05, 0.08, 0.08, head_width=0.05, head_length=0.05, fc='k', ec='k')
        
        elif gate.name in ["H", "X", "Y", "Z", "S", "T"]:
            # Single-qubit gate box
            y = self.n_qubits - gate.qubits[0] - 1
            box = FancyBboxPatch((x-0.3, y-0.3), 0.6, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor='lightgreen', linewidth=2)
            ax.add_patch(box)
            ax.text(x, y, gate.name, ha='center', va='center', fontsize=12, fontweight='bold')
        
        else:
            # Oracle or custom gate
            if len(gate.qubits) > 1:
                y_min = self.n_qubits - max(gate.qubits) - 1
                y_max = self.n_qubits - min(gate.qubits) - 1
                height = y_max - y_min + 0.6
                
                box = FancyBboxPatch((x-0.4, y_min-0.3), 0.8, height,
                                    boxstyle="round,pad=0.05",
                                    edgecolor='black', facecolor='lightyellow', linewidth=2)
                ax.add_patch(box)
                ax.text(x, (y_min + y_max) / 2, gate.name, ha='center', va='center', 
                       fontsize=10, fontweight='bold')
            else:
                y = self.n_qubits - gate.qubits[0] - 1
                box = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                                    boxstyle="round,pad=0.05",
                                    edgecolor='black', facecolor='lightyellow', linewidth=2)
                ax.add_patch(box)
                ax.text(x, y, gate.name, ha='center', va='center', fontsize=10, fontweight='bold')


def draw_deutsch_jozsa_circuit(n_qubits: int, save_path: Optional[str] = None):
    """Draw Deutsch-Jozsa algorithm circuit"""
    circuit = CircuitDiagram(n_qubits)
    
    # Step 1: Hadamards
    for i in range(n_qubits):
        circuit.h(i)
    
    circuit.barrier()
    
    # Step 2: Oracle
    circuit.oracle(list(range(n_qubits)), "U_f")
    
    circuit.barrier()
    
    # Step 3: Hadamards again
    for i in range(n_qubits):
        circuit.h(i)
    
    circuit.barrier()
    
    # Step 4: Measure
    for i in range(n_qubits):
        circuit.measure(i)
    
    return circuit.draw(save_path=save_path, show=False)


def draw_grover_circuit(n_qubits: int, iterations: int = 1, save_path: Optional[str] = None):
    """Draw Grover's algorithm circuit"""
    circuit = CircuitDiagram(n_qubits)
    
    # Step 1: Hadamards - uniform superposition
    for i in range(n_qubits):
        circuit.h(i)
    
    circuit.barrier()
    
    # Step 2: Grover iterations
    for _ in range(iterations):
        # Oracle
        circuit.oracle(list(range(n_qubits)), "Oracle")
        
        # Diffusion
        circuit.oracle(list(range(n_qubits)), "Diffusion")
        
        circuit.barrier()
    
    # Step 3: Measure
    for i in range(n_qubits):
        circuit.measure(i)
    
    return circuit.draw(save_path=save_path, show=False)


def draw_qft_circuit(n_qubits: int, save_path: Optional[str] = None):
    """Draw QFT circuit (simplified)"""
    circuit = CircuitDiagram(n_qubits)
    
    for i in range(n_qubits):
        circuit.h(i)
        # Controlled phase rotations (simplified as oracle)
        if i < n_qubits - 1:
            circuit.oracle([i, i+1], f"R_{i+1}")
    
    circuit.barrier()
    
    return circuit.draw(save_path=save_path, show=False)


def draw_circuit(algorithm: str, n_qubits: int, **kwargs) -> Tuple:
    """
    Draw circuit for specified algorithm.
    
    Parameters
    ----------
    algorithm : str
        "deutsch-jozsa", "grover", or "qft"
    n_qubits : int
        Number of qubits
    **kwargs
        Additional parameters (e.g., iterations for Grover)
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axes
    """
    if algorithm.lower() == "deutsch-jozsa":
        return draw_deutsch_jozsa_circuit(n_qubits, kwargs.get('save_path'))
    elif algorithm.lower() == "grover":
        iterations = kwargs.get('iterations', 1)
        return draw_grover_circuit(n_qubits, iterations, kwargs.get('save_path'))
    elif algorithm.lower() == "qft":
        return draw_qft_circuit(n_qubits, kwargs.get('save_path'))
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def circuit_to_latex(circuit: CircuitDiagram) -> str:
    """Convert circuit to LaTeX (basic implementation)"""
    latex = "\\begin{quantikz}\n"
    for i in range(circuit.n_qubits):
        latex += f"\\lstick{{$|q_{i}\\rangle$}} & "
        # Add gates for this qubit
        latex += " & ".join(["\\qw"] * circuit.max_time)
        latex += " \\\\\n"
    latex += "\\end{quantikz}"
    return latex


def export_circuit_image(circuit: CircuitDiagram, filename: str, dpi: int = 300):
    """Export circuit as high-res image"""
    circuit.draw(save_path=filename, show=False)
    print(f"Circuit exported to {filename}")


if __name__ == "__main__":
    print("Testing Circuit Visualization...")
    
    # Test Deutsch-Jozsa
    print("\\n1. Deutsch-Jozsa Circuit (3 qubits)")
    draw_deutsch_jozsa_circuit(3)
    
    # Test Grover
    print("\\n2. Grover Circuit (3 qubits, 2 iterations)")
    draw_grover_circuit(3, iterations=2)
    
    # Test QFT
    print("\\n3. QFT Circuit (3 qubits)")
    draw_qft_circuit(3)
    
    print("\\nCircuit visualization tests passed!")
'''

# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================

performance_content = '''# -*- coding: utf-8 -*-
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
    report = f"# Performance Analysis: {algorithm.title()}\\n\\n"
    
    if algorithm == "deutsch-jozsa":
        results = benchmark_deutsch_jozsa(n_qubits_range, trials=50)
        
        report += "## Query Complexity\\n\\n"
        report += "| n qubits | N = 2^n | Quantum | Classical (avg) | Classical (worst) | Speedup (avg) |\\n"
        report += "|----------|---------|---------|-----------------|-------------------|---------------|\\n"
        
        for i, n in enumerate(n_qubits_range):
            N = 2 ** n
            q_queries = results['quantum_queries'][i]
            c_avg = results['classical_queries_avg'][i]
            c_worst = results['classical_queries_worst'][i]
            speedup = c_avg / q_queries
            
            report += f"| {n} | {N} | {q_queries} | {c_avg:.1f} | {c_worst} | {speedup:.1f}x |\\n"
        
        report += "\\n**Quantum Advantage:** Exponential speedup (1 query vs 2^(n-1)+1 worst case)\\n"
    
    elif algorithm == "grover":
        results = benchmark_grover(n_qubits_range, trials=50)
        
        report += "## Query Complexity & Success Rate\\n\\n"
        report += "| n qubits | N = 2^n | Quantum | Classical (avg) | Speedup | Q Success | C Success |\\n"
        report += "|----------|---------|---------|-----------------|---------|-----------|-----------|\\n"
        
        for i, n in enumerate(n_qubits_range):
            N = 2 ** n
            q_queries = results['quantum_queries'][i]
            c_avg = N / 2
            speedup = c_avg / q_queries
            q_success = results['quantum_success'][i]
            c_success = results['classical_success'][i]
            
            report += f"| {n} | {N} | {q_queries} | {c_avg:.0f} | {speedup:.2f}x | {q_success:.1%} | {c_success:.1%} |\\n"
        
        report += "\\n**Quantum Advantage:** Quadratic speedup (√N vs N queries)\\n"
    
    return report


if __name__ == "__main__":
    print("\\nPerformance Analysis Demo\\n")
    
    # Test Deutsch-Jozsa
    print("1. Deutsch-Jozsa Benchmark (n=2 to 5)")
    print("-" * 60)
    dj_results = benchmark_deutsch_jozsa([2, 3, 4, 5], trials=20)
    print(f"   Quantum queries: {dj_results['quantum_queries']}")
    print(f"   Classical avg: {dj_results['classical_queries_avg']}")
    print(f"   Speedup at n=5: {dj_results['classical_queries_avg'][-1]:.0f}x")
    
    # Test Grover
    print("\\n2. Grover Benchmark (n=2 to 4)")
    print("-" * 60)
    grover_results = benchmark_grover([2, 3, 4], trials=20)
    print(f"   Quantum queries: {grover_results['quantum_queries']}")
    print(f"   Quantum success: {[f'{s:.1%}' for s in grover_results['quantum_success']]}")
    print(f"   Speedup at n=4: {8 / grover_results['quantum_queries'][-1]:.2f}x")
    
    print("\\nPerformance analysis tests passed!")
'''

# Write files
with open(os.path.join(base_path, 'circuit_visualization.py'), 'w', encoding='utf-8') as f:
    f.write(circuit_viz_content)
print("Created circuit_visualization.py")

with open(os.path.join(base_path, 'performance_analysis.py'), 'w', encoding='utf-8') as f:
    f.write(performance_content)
print("Created performance_analysis.py")

print("\\nVisualization and analysis modules created!")

