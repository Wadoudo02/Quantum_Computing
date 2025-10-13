"""
Bloch Sphere Visualization
==========================

Visualize single-qubit states on the Bloch sphere.

The Bloch sphere is a geometrical representation of qubit states:
- North pole (0, 0, 1): |0⟩
- South pole (0, 0, -1): |1⟩
- Equator: Superposition states
- Any pure state: Point on surface
- Mixed states: Points inside sphere

Reference: Imperial Notes Section 1.1 (States and Operators)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
import matplotlib.animation as animation

from .qubit import Qubit


class BlochSphere:
    """
    3D visualization of the Bloch sphere for single-qubit states.

    Examples
    --------
    >>> from .qubit import ket_0, ket_plus
    >>> from .gates import hadamard
    >>>
    >>> bloch = BlochSphere()
    >>> bloch.add_qubit(ket_0(), label="|0⟩", color='blue')
    >>> bloch.add_qubit(ket_plus(), label="|+⟩", color='green')
    >>> bloch.show()
    """

    def __init__(self, figsize=(10, 10)):
        """
        Initialize Bloch sphere visualization.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches
        """
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.qubits = []
        self.labels = []
        self.colors = []
        self.vectors = []

    def _draw_sphere(self):
        """Draw the Bloch sphere surface."""
        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot sphere with transparency
        self.ax.plot_surface(x, y, z, color='cyan', alpha=0.1,
                            linewidth=0, antialiased=True)

    def _draw_axes(self):
        """Draw coordinate axes and labels."""
        # Draw axes
        axis_length = 1.3

        # X axis (red)
        self.ax.plot([-axis_length, axis_length], [0, 0], [0, 0],
                    'r-', linewidth=2, alpha=0.6)
        self.ax.text(axis_length + 0.1, 0, 0, 'X', fontsize=14, color='red')

        # Y axis (green)
        self.ax.plot([0, 0], [-axis_length, axis_length], [0, 0],
                    'g-', linewidth=2, alpha=0.6)
        self.ax.text(0, axis_length + 0.1, 0, 'Y', fontsize=14, color='green')

        # Z axis (blue)
        self.ax.plot([0, 0], [0, 0], [-axis_length, axis_length],
                    'b-', linewidth=2, alpha=0.6)
        self.ax.text(0, 0, axis_length + 0.1, 'Z', fontsize=14, color='blue')

    def _draw_equator_and_meridians(self):
        """Draw equator and meridian circles."""
        theta = np.linspace(0, 2 * np.pi, 100)

        # Equator (XY plane)
        x_eq = np.cos(theta)
        y_eq = np.sin(theta)
        z_eq = np.zeros_like(theta)
        self.ax.plot(x_eq, y_eq, z_eq, 'k--', linewidth=1, alpha=0.3)

        # Meridian (XZ plane)
        x_mer = np.cos(theta)
        z_mer = np.sin(theta)
        y_mer = np.zeros_like(theta)
        self.ax.plot(x_mer, y_mer, z_mer, 'k--', linewidth=1, alpha=0.3)

        # Meridian (YZ plane)
        y_mer2 = np.cos(theta)
        z_mer2 = np.sin(theta)
        x_mer2 = np.zeros_like(theta)
        self.ax.plot(x_mer2, y_mer2, z_mer2, 'k--', linewidth=1, alpha=0.3)

    def _draw_basis_states(self):
        """Draw and label basis states."""
        # |0⟩ at north pole
        self.ax.scatter([0], [0], [1], color='blue', s=100, alpha=0.7)
        self.ax.text(0, 0, 1.15, '|0⟩', fontsize=12, ha='center')

        # |1⟩ at south pole
        self.ax.scatter([0], [0], [-1], color='blue', s=100, alpha=0.7)
        self.ax.text(0, 0, -1.15, '|1⟩', fontsize=12, ha='center')

        # |+⟩ on X axis
        self.ax.scatter([1], [0], [0], color='green', s=80, alpha=0.7)
        self.ax.text(1.15, 0, 0, '|+⟩', fontsize=11, ha='center')

        # |−⟩ on -X axis
        self.ax.scatter([-1], [0], [0], color='green', s=80, alpha=0.7)
        self.ax.text(-1.15, 0, 0, '|−⟩', fontsize=11, ha='center')

        # |+i⟩ on Y axis
        self.ax.scatter([0], [1], [0], color='purple', s=80, alpha=0.7)
        self.ax.text(0, 1.15, 0, '|+i⟩', fontsize=11, ha='center')

        # |−i⟩ on -Y axis
        self.ax.scatter([0], [-1], [0], color='purple', s=80, alpha=0.7)
        self.ax.text(0, -1.15, 0, '|−i⟩', fontsize=11, ha='center')

    def add_qubit(self, qubit: Qubit, label: Optional[str] = None,
                  color: str = 'red', show_vector: bool = True):
        """
        Add a qubit state to the visualization.

        Parameters
        ----------
        qubit : Qubit
            Qubit to visualize
        label : str, optional
            Label for the state
        color : str, optional
            Color for the point and vector (default: 'red')
        show_vector : bool, optional
            Whether to draw vector from origin (default: True)
        """
        self.qubits.append(qubit)
        self.labels.append(label)
        self.colors.append(color)
        self.vectors.append(show_vector)

    def add_qubits(self, qubits: List[Qubit], labels: Optional[List[str]] = None,
                   colors: Optional[List[str]] = None):
        """
        Add multiple qubit states at once.

        Parameters
        ----------
        qubits : list of Qubit
            Qubits to visualize
        labels : list of str, optional
            Labels for each state
        colors : list of str, optional
            Colors for each state
        """
        if labels is None:
            labels = [None] * len(qubits)
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(qubits)))

        for qubit, label, color in zip(qubits, labels, colors):
            self.add_qubit(qubit, label, color)

    def _plot_qubits(self):
        """Plot all added qubits."""
        for qubit, label, color, show_vec in zip(self.qubits, self.labels,
                                                   self.colors, self.vectors):
            x, y, z = qubit.bloch_coordinates()

            # Plot point
            self.ax.scatter([x], [y], [z], color=color, s=200,
                          edgecolors='black', linewidth=2, alpha=0.9)

            # Draw vector from origin
            if show_vec:
                self.ax.quiver(0, 0, 0, x, y, z, color=color,
                             arrow_length_ratio=0.15, linewidth=2.5, alpha=0.8)

            # Add label
            if label:
                # Offset label slightly from point
                offset = 0.15
                self.ax.text(x + offset, y + offset, z + offset, label,
                           fontsize=12, color=color, fontweight='bold')

    def show(self, title: str = "Bloch Sphere", elev: float = 20, azim: float = 45):
        """
        Display the Bloch sphere.

        Parameters
        ----------
        title : str, optional
            Title for the plot
        elev : float, optional
            Elevation angle for view (default: 20)
        azim : float, optional
            Azimuthal angle for view (default: 45)
        """
        self._draw_sphere()
        self._draw_axes()
        self._draw_equator_and_meridians()
        self._draw_basis_states()
        self._plot_qubits()

        # Set labels and title
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.set_zlabel('Z', fontsize=12)
        self.ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])

        # Set view angle
        self.ax.view_init(elev=elev, azim=azim)

        # Set axis limits
        self.ax.set_xlim([-1.3, 1.3])
        self.ax.set_ylim([-1.3, 1.3])
        self.ax.set_zlim([-1.3, 1.3])

        # Remove grid
        self.ax.grid(False)

        plt.tight_layout()
        plt.show()

    def save(self, filename: str, title: str = "Bloch Sphere",
             elev: float = 20, azim: float = 45, dpi: int = 300):
        """
        Save the Bloch sphere to a file.

        Parameters
        ----------
        filename : str
            Output filename (e.g., 'bloch.png')
        title : str, optional
            Title for the plot
        elev : float, optional
            Elevation angle for view
        azim : float, optional
            Azimuthal angle for view
        dpi : int, optional
            Resolution in dots per inch (default: 300)
        """
        self._draw_sphere()
        self._draw_axes()
        self._draw_equator_and_meridians()
        self._draw_basis_states()
        self._plot_qubits()

        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.set_zlabel('Z', fontsize=12)
        self.ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_xlim([-1.3, 1.3])
        self.ax.set_ylim([-1.3, 1.3])
        self.ax.set_zlim([-1.3, 1.3])
        self.ax.grid(False)

        plt.tight_layout()
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Saved to {filename}")


def plot_gate_trajectory(initial_qubit: Qubit, gates: List, gate_names: List[str],
                         title: str = "Gate Operations on Bloch Sphere"):
    """
    Visualize the trajectory of a qubit under a sequence of gates.

    Parameters
    ----------
    initial_qubit : Qubit
        Starting qubit state
    gates : list
        List of gate matrices to apply
    gate_names : list of str
        Names of the gates for labeling
    title : str, optional
        Title for the plot

    Examples
    --------
    >>> from .qubit import ket_0
    >>> from .gates import HADAMARD, PAULI_X, PAULI_Z
    >>>
    >>> plot_gate_trajectory(
    ...     ket_0(),
    ...     [HADAMARD, PAULI_Z, PAULI_X],
    ...     ['H', 'Z', 'X']
    ... )
    """
    from .gates import apply_gate

    # Initialize
    bloch = BlochSphere(figsize=(12, 10))
    current_qubit = initial_qubit.copy()

    # Plot initial state
    bloch.add_qubit(current_qubit, label='Start', color='green')

    # Apply gates and plot trajectory
    colors = plt.cm.rainbow(np.linspace(0, 1, len(gates)))
    trajectory_points = [current_qubit.bloch_coordinates()]

    for i, (gate, name) in enumerate(zip(gates, gate_names)):
        current_qubit = apply_gate(current_qubit, gate)
        bloch.add_qubit(current_qubit, label=name, color=colors[i])
        trajectory_points.append(current_qubit.bloch_coordinates())

    # Draw trajectory line
    trajectory = np.array(trajectory_points)
    bloch.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                 'k--', linewidth=2, alpha=0.5, label='Trajectory')

    bloch.show(title=title)


def compare_states(qubits: List[Qubit], labels: List[str],
                   title: str = "Qubit State Comparison"):
    """
    Compare multiple qubit states on the Bloch sphere.

    Parameters
    ----------
    qubits : list of Qubit
        Qubit states to compare
    labels : list of str
        Labels for each state
    title : str, optional
        Title for the plot

    Examples
    --------
    >>> from .qubit import ket_0, ket_1, ket_plus, ket_minus
    >>>
    >>> compare_states(
    ...     [ket_0(), ket_1(), ket_plus(), ket_minus()],
    ...     ['|0⟩', '|1⟩', '|+⟩', '|−⟩']
    ... )
    """
    bloch = BlochSphere(figsize=(12, 10))
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

    for i, (qubit, label) in enumerate(zip(qubits, labels)):
        color = colors[i % len(colors)]
        bloch.add_qubit(qubit, label=label, color=color)

    bloch.show(title=title)


def animate_rotation(initial_qubit: Qubit, gate, n_steps: int = 50,
                     save_as: Optional[str] = None):
    """
    Animate the continuous rotation of a qubit state under a gate.

    Parameters
    ----------
    initial_qubit : Qubit
        Starting qubit state
    gate : np.ndarray
        Gate matrix to apply incrementally
    n_steps : int, optional
        Number of animation frames (default: 50)
    save_as : str, optional
        If provided, save animation to this file (e.g., 'rotation.gif')

    Examples
    --------
    >>> from .qubit import ket_0
    >>> from .gates import PAULI_X
    >>>
    >>> animate_rotation(ket_0(), PAULI_X, n_steps=30)
    """
    from .gates import apply_gate
    from scipy.linalg import fractional_matrix_power

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Generate intermediate gates (fractional powers)
    qubits = []
    for i in range(n_steps + 1):
        fraction = i / n_steps
        partial_gate = fractional_matrix_power(gate, fraction)
        qubits.append(apply_gate(initial_qubit, partial_gate))

    def update(frame):
        ax.clear()

        # Create temporary BlochSphere for drawing
        temp_bloch = BlochSphere(figsize=(10, 10))
        temp_bloch.ax = ax
        temp_bloch.fig = fig

        temp_bloch._draw_sphere()
        temp_bloch._draw_axes()
        temp_bloch._draw_equator_and_meridians()
        temp_bloch._draw_basis_states()

        # Plot current position
        x, y, z = qubits[frame].bloch_coordinates()
        ax.scatter([x], [y], [z], color='red', s=200, edgecolors='black', linewidth=2)
        ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.15, linewidth=2.5)

        # Plot trajectory so far
        if frame > 0:
            trajectory = np.array([q.bloch_coordinates() for q in qubits[:frame+1]])
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   'b-', linewidth=2, alpha=0.6)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Gate Animation (Step {frame}/{n_steps})', fontsize=14)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=45)
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_zlim([-1.3, 1.3])
        ax.grid(False)

    anim = animation.FuncAnimation(fig, update, frames=n_steps+1,
                                  interval=50, repeat=True)

    if save_as:
        anim.save(save_as, writer='pillow', fps=20)
        print(f"Animation saved to {save_as}")

    plt.show()
    return anim
