"""
Visualizations for Quantum Error Correction

This module provides visualization tools for error correction concepts:
- Circuit diagrams for encoding, syndrome measurement, and recovery
- Error rate curves and threshold plots
- Syndrome measurement results
- Success rate comparisons
- Overhead analysis plots

Author: Quantum Computing Learning Project
Phase: 5 - Error Correction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class ECCVisualizer:
    """
    Visualizer for quantum error correction concepts.
    """

    def __init__(self, style: str = 'default'):
        """
        Initialize the visualizer.

        Args:
            style: Matplotlib style to use
        """
        self.style = style
        self.colors = {
            'physical': '#3498db',
            'logical': '#e74c3c',
            'threshold': '#2ecc71',
            'success': '#27ae60',
            'error': '#e74c3c',
            'syndrome': '#f39c12',
            'ancilla': '#9b59b6'
        }

    def plot_circuit(
        self,
        circuit: QuantumCircuit,
        title: str = "Quantum Circuit",
        save_path: Optional[str] = None
    ):
        """
        Plot a quantum circuit.

        Args:
            circuit: QuantumCircuit to plot
            title: Title for the plot
            save_path: Optional path to save the figure
        """
        fig = circuit.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})
        plt.suptitle(title, fontsize=14, y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def plot_error_rates(
        self,
        physical_rates: np.ndarray,
        logical_rates: Dict[str, np.ndarray],
        title: str = "Logical vs Physical Error Rates",
        save_path: Optional[str] = None
    ):
        """
        Plot logical error rates vs physical error rates for multiple codes.

        Args:
            physical_rates: Array of physical error rates
            logical_rates: Dictionary of {code_name: logical_error_rates}
            title: Plot title
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot y=x line (no error correction)
        ax.plot(physical_rates, physical_rates, 'k--', linewidth=2,
                label='No Correction', alpha=0.7)

        # Plot each code
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        markers = ['o', 's', '^', 'D', 'v']

        for i, (name, rates) in enumerate(logical_rates.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            ax.plot(physical_rates, rates, marker=marker, linewidth=2,
                   markersize=8, label=name, color=color, alpha=0.8)

        ax.set_xlabel('Physical Error Rate', fontsize=12)
        ax.set_ylabel('Logical Error Rate', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, shadow=True)

        # Add shaded region for sub-threshold
        ax.axhspan(physical_rates.min(), physical_rates.max(),
                  alpha=0.1, color='green', label='Sub-threshold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_threshold_analysis(
        self,
        physical_rates: np.ndarray,
        logical_rates: np.ndarray,
        code_name: str,
        threshold: Optional[float] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot error threshold analysis for a single code.

        Args:
            physical_rates: Array of physical error rates
            logical_rates: Array of logical error rates
            code_name: Name of the code
            threshold: Threshold error rate (if known)
            save_path: Optional path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Log-log plot
        ax1.plot(physical_rates, physical_rates, 'k--', linewidth=2,
                label='No Correction', alpha=0.7)
        ax1.plot(physical_rates, logical_rates, 'o-', linewidth=2,
                markersize=8, label=code_name, color=self.colors['physical'])

        if threshold:
            ax1.axvline(threshold, color=self.colors['threshold'],
                       linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')

        ax1.set_xlabel('Physical Error Rate', fontsize=12)
        ax1.set_ylabel('Logical Error Rate', fontsize=12)
        ax1.set_title(f'{code_name} - Log Scale', fontsize=14)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Right plot: Linear scale (zoomed)
        ax2.plot(physical_rates, physical_rates, 'k--', linewidth=2,
                label='No Correction', alpha=0.7)
        ax2.plot(physical_rates, logical_rates, 'o-', linewidth=2,
                markersize=8, label=code_name, color=self.colors['physical'])

        if threshold:
            ax2.axvline(threshold, color=self.colors['threshold'],
                       linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')

        ax2.set_xlabel('Physical Error Rate', fontsize=12)
        ax2.set_ylabel('Logical Error Rate', fontsize=12)
        ax2.set_title(f'{code_name} - Linear Scale', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.suptitle(f'Threshold Analysis: {code_name}', fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_syndrome_distribution(
        self,
        syndrome_counts: Dict[str, int],
        title: str = "Syndrome Distribution",
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of syndrome measurements.

        Args:
            syndrome_counts: Dictionary of {syndrome: count}
            title: Plot title
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        syndromes = list(syndrome_counts.keys())
        counts = list(syndrome_counts.values())
        total = sum(counts)
        percentages = [c / total * 100 for c in counts]

        bars = ax.bar(syndromes, percentages, color=self.colors['syndrome'],
                     alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Syndrome', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_success_rates(
        self,
        codes: Dict[str, List[Tuple[float, float]]],
        title: str = "Error Correction Success Rates",
        save_path: Optional[str] = None
    ):
        """
        Plot success rates for different codes at various error rates.

        Args:
            codes: Dictionary of {code_name: [(error_rate, success_rate), ...]}
            title: Plot title
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        markers = ['o', 's', '^', 'D', 'v']

        for i, (name, data) in enumerate(codes.items()):
            error_rates = [d[0] for d in data]
            success_rates = [d[1] * 100 for d in data]  # Convert to percentage

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            ax.plot(error_rates, success_rates, marker=marker, linewidth=2,
                   markersize=8, label=name, color=color, alpha=0.8)

        ax.set_xlabel('Physical Error Rate', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.set_ylim([0, 105])

        # Add reference lines
        ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50%')
        ax.axhline(90, color='green', linestyle='--', alpha=0.5, label='90%')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_overhead_comparison(
        self,
        codes: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """
        Plot overhead comparison between different codes.

        Args:
            codes: Dictionary with overhead metrics per code
            save_path: Optional path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        code_names = list(codes.keys())
        physical_qubits = [codes[name]['physical_qubits'] for name in code_names]
        overheads = [codes[name]['overhead'] for name in code_names]

        colors_list = [self.colors['physical'], self.colors['logical'],
                      self.colors['threshold'], self.colors['syndrome']]

        # Left plot: Physical qubits
        bars1 = ax1.bar(code_names, physical_qubits, color=colors_list[:len(code_names)],
                       alpha=0.7, edgecolor='black', linewidth=1.5)

        for bar, val in zip(bars1, physical_qubits):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax1.set_ylabel('Physical Qubits', fontsize=12)
        ax1.set_title('Physical Qubit Requirements', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')

        # Right plot: Overhead
        bars2 = ax2.bar(code_names, overheads, color=colors_list[:len(code_names)],
                       alpha=0.7, edgecolor='black', linewidth=1.5)

        for bar, val in zip(bars2, overheads):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}x',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2.set_ylabel('Overhead (Physical/Logical)', fontsize=12)
        ax2.set_title('Encoding Overhead', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Code Overhead Comparison', fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_error_suppression(
        self,
        error_rates: np.ndarray,
        suppression_factors: Dict[str, np.ndarray],
        title: str = "Error Suppression Factor",
        save_path: Optional[str] = None
    ):
        """
        Plot error suppression factor (p_physical / p_logical).

        Args:
            error_rates: Array of physical error rates
            suppression_factors: Dictionary of {code_name: suppression_factors}
            title: Plot title
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

        for i, (name, factors) in enumerate(suppression_factors.items()):
            color = colors[i % len(colors)]
            ax.plot(error_rates, factors, linewidth=2.5,
                   label=name, color=color, alpha=0.8)

        ax.axhline(1, color='black', linestyle='--', linewidth=2,
                  label='No Improvement', alpha=0.7)

        ax.set_xlabel('Physical Error Rate', fontsize=12)
        ax.set_ylabel('Suppression Factor (P_phys / P_logical)', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, shadow=True)

        # Shade region where suppression > 1 (improvement)
        ax.fill_between(error_rates, 1, ax.get_ylim()[1],
                       alpha=0.1, color='green', label='Improvement Region')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_gates_vs_error_rate(
        self,
        error_rates: np.ndarray,
        gates_before_failure: Dict[str, np.ndarray],
        title: str = "Algorithm Depth vs Error Rate",
        save_path: Optional[str] = None
    ):
        """
        Plot number of gates achievable vs error rate.

        Args:
            error_rates: Array of error rates
            gates_before_failure: Dictionary of {code_name: gates_array}
            title: Plot title
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

        for i, (name, gates) in enumerate(gates_before_failure.items()):
            color = colors[i % len(colors)]
            ax.plot(error_rates, gates, linewidth=2.5,
                   label=name, color=color, alpha=0.8)

        ax.set_xlabel('Error Rate', fontsize=12)
        ax.set_ylabel('Gates Before Failure (50% threshold)', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, shadow=True)

        # Add reference lines for algorithm requirements
        ax.axhline(1000, color='gray', linestyle='--', alpha=0.5)
        ax.text(error_rates[-1], 1100, 'Small algorithms',
               ha='right', fontsize=10, alpha=0.7)

        ax.axhline(10000, color='gray', linestyle='--', alpha=0.5)
        ax.text(error_rates[-1], 11000, 'Medium algorithms',
               ha='right', fontsize=10, alpha=0.7)

        ax.axhline(100000, color='gray', linestyle='--', alpha=0.5)
        ax.text(error_rates[-1], 110000, 'Large algorithms',
               ha='right', fontsize=10, alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def create_encoding_diagram(
        self,
        n_physical: int,
        n_logical: int,
        code_name: str,
        save_path: Optional[str] = None
    ):
        """
        Create a conceptual diagram of encoding.

        Args:
            n_physical: Number of physical qubits
            n_logical: Number of logical qubits
            code_name: Name of the code
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Logical qubit
        logical_box = FancyBboxPatch((1, 2), 2, 1.5, boxstyle="round,pad=0.1",
                                     edgecolor=self.colors['logical'],
                                     facecolor=self.colors['logical'],
                                     alpha=0.3, linewidth=3)
        ax.add_patch(logical_box)
        ax.text(2, 2.75, f'{n_logical} Logical\nQubit(s)',
               ha='center', va='center', fontsize=12, fontweight='bold')

        # Arrow
        ax.annotate('', xy=(5.5, 2.75), xytext=(3.5, 2.75),
                   arrowprops=dict(arrowstyle='->', lw=3, color='black'))
        ax.text(4.5, 3.2, 'Encoding', ha='center', fontsize=11, fontweight='bold')

        # Physical qubits
        qubit_spacing = 0.4
        start_y = 2.75 - (n_physical * qubit_spacing) / 2

        for i in range(n_physical):
            y_pos = start_y + i * qubit_spacing
            circle = Circle((7, y_pos), 0.15, edgecolor=self.colors['physical'],
                          facecolor=self.colors['physical'], alpha=0.6, linewidth=2)
            ax.add_patch(circle)
            ax.text(7.5, y_pos, f'q{i}', va='center', fontsize=9)

        # Physical qubits box
        physical_box = FancyBboxPatch((6, start_y - 0.3), 2, n_physical * qubit_spacing + 0.3,
                                     boxstyle="round,pad=0.1",
                                     edgecolor=self.colors['physical'],
                                     facecolor='none',
                                     alpha=0.5, linewidth=2, linestyle='--')
        ax.add_patch(physical_box)
        ax.text(7, start_y - 0.6, f'{n_physical} Physical Qubits',
               ha='center', fontsize=11, fontweight='bold')

        ax.set_xlim(0, 9)
        ax.set_ylim(1, 5)
        ax.axis('off')
        ax.set_title(f'{code_name}: Encoding Overview', fontsize=14, pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def demo_visualizations():
    """Demonstrate visualization capabilities."""
    print("Generating Phase 5 visualizations...")

    viz = ECCVisualizer()

    # 1. Error rates comparison
    physical_rates = np.logspace(-4, -1, 20)

    from error_analysis import ErrorAnalyzer
    analyzer = ErrorAnalyzer()

    logical_rates = {
        '3-Qubit': np.array([analyzer.compute_three_qubit_logical_error(p)
                            for p in physical_rates]),
        'Shor (9-qubit)': np.array([analyzer.compute_shor_code_logical_error(p)
                                    for p in physical_rates]),
        '5-Qubit': np.array([analyzer.compute_five_qubit_logical_error(p)
                            for p in physical_rates]),
    }

    viz.plot_error_rates(physical_rates, logical_rates,
                        title="Logical vs Physical Error Rates")

    print("✓ Generated error rates plot")

    # 2. Overhead comparison
    codes_overhead = {
        '3-Qubit': {'physical_qubits': 3, 'logical_qubits': 1, 'overhead': 3.0},
        '5-Qubit': {'physical_qubits': 5, 'logical_qubits': 1, 'overhead': 5.0},
        'Shor': {'physical_qubits': 9, 'logical_qubits': 1, 'overhead': 9.0},
    }

    viz.plot_overhead_comparison(codes_overhead)

    print("✓ Generated overhead comparison")

    # 3. Encoding diagram
    viz.create_encoding_diagram(9, 1, "Shor's 9-Qubit Code")

    print("✓ Generated encoding diagram")

    print("\nVisualization demo complete!")


if __name__ == "__main__":
    demo_visualizations()
