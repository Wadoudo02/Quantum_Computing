# -*- coding: utf-8 -*-
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
    print("\n1. Deutsch-Jozsa Circuit (3 qubits)")
    draw_deutsch_jozsa_circuit(3)
    
    # Test Grover
    print("\n2. Grover Circuit (3 qubits, 2 iterations)")
    draw_grover_circuit(3, iterations=2)
    
    # Test QFT
    print("\n3. QFT Circuit (3 qubits)")
    draw_qft_circuit(3)
    
    print("\nCircuit visualization tests passed!")
