"""
Analysis Tools Module for Phase 6: Visualization & Comparison

This module provides tools for analyzing and visualizing quantum hardware
performance, comparing backends, and presenting results.

Author: Wadoud Charbak
Date: November 2024
For: Quantinuum & Riverlane Recruitment
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase6_hardware.hardware_interface import HardwareSpecs, get_backend_specs
from phase6_hardware.noise_models import RealisticNoiseModel


def plot_backend_comparison(
    backend_names: List[str],
    save_path: Optional[str] = None
):
    """
    Create comprehensive comparison plot of quantum backends.

    Args:
        backend_names: List of backend names to compare
        save_path: Optional path to save figure
    """
    specs_list = [get_backend_specs(name) for name in backend_names]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Quantum Hardware Backend Comparison', fontsize=16, fontweight='bold')

    # 1. Number of qubits
    ax = axes[0, 0]
    qubits = [spec.num_qubits for spec in specs_list]
    ax.bar(range(len(backend_names)), qubits, color='skyblue', edgecolor='black')
    ax.set_xticks(range(len(backend_names)))
    ax.set_xticklabels(backend_names, rotation=45, ha='right')
    ax.set_ylabel('Number of Qubits')
    ax.set_title('System Size')
    ax.grid(axis='y', alpha=0.3)

    # 2. T1 times
    ax = axes[0, 1]
    t1_avg = [np.mean(spec.t1_times) for spec in specs_list]
    t1_std = [np.std(spec.t1_times) for spec in specs_list]
    ax.bar(range(len(backend_names)), t1_avg, yerr=t1_std,
           color='lightgreen', edgecolor='black', capsize=5)
    ax.set_xticks(range(len(backend_names)))
    ax.set_xticklabels(backend_names, rotation=45, ha='right')
    ax.set_ylabel('T1 (μs)')
    ax.set_title('Energy Relaxation Time (T1)')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    # 3. T2 times
    ax = axes[0, 2]
    t2_avg = [np.mean(spec.t2_times) for spec in specs_list]
    t2_std = [np.std(spec.t2_times) for spec in specs_list]
    ax.bar(range(len(backend_names)), t2_avg, yerr=t2_std,
           color='lightcoral', edgecolor='black', capsize=5)
    ax.set_xticks(range(len(backend_names)))
    ax.set_xticklabels(backend_names, rotation=45, ha='right')
    ax.set_ylabel('T2 (μs)')
    ax.set_title('Dephasing Time (T2)')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    # 4. Gate fidelities
    ax = axes[1, 0]
    x = np.arange(len(backend_names))
    width = 0.35

    sq_fidelities = [spec.gate_fidelities.get('single_qubit', 1.0) for spec in specs_list]
    tq_fidelities = [spec.gate_fidelities.get('cnot', 1.0) for spec in specs_list]

    ax.bar(x - width/2, sq_fidelities, width, label='Single-Qubit', color='gold', edgecolor='black')
    ax.bar(x + width/2, tq_fidelities, width, label='Two-Qubit', color='orange', edgecolor='black')

    ax.set_ylabel('Gate Fidelity')
    ax.set_title('Gate Fidelities')
    ax.set_xticks(x)
    ax.set_xticklabels(backend_names, rotation=45, ha='right')
    ax.set_ylim([0.8, 1.0])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 5. Readout fidelity
    ax = axes[1, 1]
    readout_fidelities = [spec.readout_fidelity for spec in specs_list]
    ax.bar(range(len(backend_names)), readout_fidelities,
           color='mediumpurple', edgecolor='black')
    ax.set_xticks(range(len(backend_names)))
    ax.set_xticklabels(backend_names, rotation=45, ha='right')
    ax.set_ylabel('Readout Fidelity')
    ax.set_title('Measurement Fidelity')
    ax.set_ylim([0.8, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # 6. Connectivity
    ax = axes[1, 2]
    connectivity = [spec.connectivity_degree() for spec in specs_list]
    ax.bar(range(len(backend_names)), connectivity,
           color='lightblue', edgecolor='black')
    ax.set_xticks(range(len(backend_names)))
    ax.set_xticklabels(backend_names, rotation=45, ha='right')
    ax.set_ylabel('Connectivity Ratio')
    ax.set_title('Qubit Connectivity')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_circuit_fidelity_vs_depth(
    backend_names: List[str],
    max_depth: int = 100,
    save_path: Optional[str] = None
):
    """
    Plot circuit fidelity vs depth for different backends.

    Args:
        backend_names: List of backend names
        max_depth: Maximum circuit depth
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    depths = np.arange(1, max_depth + 1)

    for backend_name in backend_names:
        noise_model = RealisticNoiseModel(get_backend_specs(backend_name))

        fidelities = []
        for depth in depths:
            # Assume circuit has roughly equal single and two-qubit gates
            num_sq = depth
            num_tq = depth // 2

            fidelity = noise_model.estimate_circuit_fidelity(num_sq, num_tq)
            fidelities.append(fidelity)

        ax.plot(depths, fidelities, marker='o', markersize=3,
                label=backend_name, linewidth=2)

    ax.set_xlabel('Circuit Depth', fontsize=12)
    ax.set_ylabel('Circuit Fidelity', fontsize=12)
    ax.set_title('Circuit Fidelity vs Depth Across Backends', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add reference lines
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_error_mitigation_comparison(
    ideal_value: float,
    noisy_value: float,
    mitigated_values: Dict[str, float],
    save_path: Optional[str] = None
):
    """
    Plot comparison of error mitigation techniques.

    Args:
        ideal_value: True ideal expectation value
        noisy_value: Unmitigated noisy value
        mitigated_values: Dictionary of {method: value}
        save_path: Optional save path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Absolute values
    methods = ['Ideal', 'Noisy'] + list(mitigated_values.keys())
    values = [ideal_value, noisy_value] + list(mitigated_values.values())
    colors = ['green', 'red'] + ['blue'] * len(mitigated_values)

    ax1.bar(range(len(methods)), values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_ylabel('Expectation Value')
    ax1.set_title('Expectation Values: Ideal vs Noisy vs Mitigated')
    ax1.axhline(y=ideal_value, color='green', linestyle='--', alpha=0.5, label='Ideal')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()

    # Subplot 2: Error reduction
    noisy_error = abs(noisy_value - ideal_value)
    errors = [0, noisy_error]  # Ideal has no error, noisy has full error

    for method, value in mitigated_values.items():
        error = abs(value - ideal_value)
        errors.append(error)

    ax2.bar(range(len(methods)), errors, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Error Comparison')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_connectivity_graph(
    backend_name: str,
    save_path: Optional[str] = None
):
    """
    Plot qubit connectivity graph for a backend.

    Args:
        backend_name: Name of backend
        save_path: Optional save path
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX not installed. Install with: pip install networkx")
        return None

    specs = get_backend_specs(backend_name)

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(specs.num_qubits))
    G.add_edges_from(specs.connectivity)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, ax=ax)

    ax.set_title(f'Qubit Connectivity Graph: {backend_name}',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_benchmark_summary_table(
    benchmarks: Dict[str, Dict[str, float]]
) -> str:
    """
    Create formatted table of benchmark results.

    Args:
        benchmarks: Dictionary of {backend: {metric: value}}

    Returns:
        Formatted table string
    """
    # Extract metrics
    all_metrics = set()
    for backend_benchmarks in benchmarks.values():
        all_metrics.update(backend_benchmarks.keys())

    all_metrics = sorted(list(all_metrics))

    # Create table
    table = "=" * 80 + "\n"
    table += "QUANTUM HARDWARE BENCHMARK SUMMARY\n"
    table += "=" * 80 + "\n\n"

    # Header
    table += f"{'Backend':<25}"
    for metric in all_metrics:
        table += f"{metric:<20}"
    table += "\n"
    table += "-" * 80 + "\n"

    # Data
    for backend, metrics in benchmarks.items():
        table += f"{backend:<25}"
        for metric in all_metrics:
            value = metrics.get(metric, 0.0)
            if isinstance(value, float):
                table += f"{value:<20.6f}"
            else:
                table += f"{value:<20}"
        table += "\n"

    table += "=" * 80 + "\n"

    return table


def plot_decoherence_curves(
    t1: float,
    t2: float,
    max_time: float = 300,
    save_path: Optional[str] = None
):
    """
    Plot T1 and T2 decoherence curves.

    Args:
        t1: T1 time in microseconds
        t2: T2 time in microseconds
        max_time: Maximum time to plot
        save_path: Optional save path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    times = np.linspace(0, max_time, 1000)

    # T1 decay
    p1 = np.exp(-times / t1)
    ax1.plot(times, p1, 'b-', linewidth=2, label=f'T1 = {t1:.1f} μs')
    ax1.fill_between(times, 0, p1, alpha=0.3)
    ax1.set_xlabel('Time (μs)', fontsize=12)
    ax1.set_ylabel('Population of |1⟩', fontsize=12)
    ax1.set_title('T1 (Energy Relaxation) Decay', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1])

    # T2 decay
    p2 = np.exp(-times / t2)
    ax2.plot(times, p2, 'r-', linewidth=2, label=f'T2 = {t2:.1f} μs')
    ax2.fill_between(times, 0, p2, alpha=0.3, color='red')
    ax2.set_xlabel('Time (μs)', fontsize=12)
    ax2.set_ylabel('Coherence', fontsize=12)
    ax2.set_title('T2 (Dephasing) Decay', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1])

    # Add gate time references
    gate_times = {'1Q': 0.05, '2Q': 0.3}  # μs
    for ax in [ax1, ax2]:
        for gate, time in gate_times.items():
            ax.axvline(x=time, color='green', linestyle='--', alpha=0.5)
            ax.text(time, 0.9, f'{gate} gate', rotation=90, va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Demonstrations
    print("=" * 70)
    print("ANALYSIS TOOLS DEMONSTRATION")
    print("=" * 70)

    # 1. Backend comparison plot
    print("\nGenerating backend comparison plot...")
    backends = ['ibm_jakarta', 'ionq_harmony', 'rigetti_aspen_m3']
    fig1 = plot_backend_comparison(backends)
    print("✓ Backend comparison plot created")

    # 2. Circuit fidelity vs depth
    print("\nGenerating circuit fidelity vs depth plot...")
    fig2 = plot_circuit_fidelity_vs_depth(backends, max_depth=50)
    print("✓ Fidelity vs depth plot created")

    # 3. Error mitigation comparison
    print("\nGenerating error mitigation comparison plot...")
    ideal = 1.0
    noisy = 0.7
    mitigated = {
        'Readout': 0.85,
        'ZNE': 0.90,
        'PEC': 0.95
    }
    fig3 = plot_error_mitigation_comparison(ideal, noisy, mitigated)
    print("✓ Error mitigation comparison plot created")

    # 4. Decoherence curves
    print("\nGenerating decoherence curves...")
    fig4 = plot_decoherence_curves(t1=100, t2=75)
    print("✓ Decoherence curves created")

    # 5. Benchmark table
    print("\nGenerating benchmark summary table...")
    benchmarks = {
        'IBM Jakarta': {
            'Avg Gate Fidelity': 0.9950,
            'T1 (μs)': 138.7,
            'T2 (μs)': 91.2,
            'Quantum Volume': 32
        },
        'IonQ Harmony': {
            'Avg Gate Fidelity': 0.9985,
            'T1 (μs)': 1000000,
            'T2 (μs)': 500000,
            'Quantum Volume': 256
        }
    }
    table = create_benchmark_summary_table(benchmarks)
    print(table)

    plt.show()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("• Trapped ions have much longer coherence times")
    print("• Superconducting qubits have better connectivity (for some)")
    print("• Circuit fidelity degrades exponentially with depth")
    print("• Error mitigation can recover 2-10x improvement")
    print("=" * 70)
