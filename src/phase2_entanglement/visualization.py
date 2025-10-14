"""
Visualization Tools for Phase 2
================================

High-quality visualizations for entanglement and Bell's inequality,
including static plots and animations suitable for LinkedIn posts.

Features:
- Density matrix heatmaps
- Correlation plots
- CHSH inequality demonstrations
- Animated violation demonstrations
- Publication-quality figures

Reference: Based on concepts from the section on Bell's inequality
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib import cm
from typing import Optional, Tuple, List
from pathlib import Path

# Handle both package and direct imports
try:
    from .bell_states import BellState, bell_phi_plus
    from .bells_inequality import (
        compute_chsh_value,
        demonstrate_bell_violation,
        scan_chsh_angles,
        optimal_chsh_angles,
        classical_bound,
        quantum_bound
    )
except ImportError:
    from bell_states import BellState, bell_phi_plus
    from bells_inequality import (
        compute_chsh_value,
        demonstrate_bell_violation,
        scan_chsh_angles,
        optimal_chsh_angles,
        classical_bound,
        quantum_bound
    )


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (10, 8)


def plot_density_matrix(
    state: BellState,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap visualization of the density matrix.

    Parameters
    ----------
    state : BellState
        The quantum state to visualize
    title : str, optional
        Plot title
    save_path : str, optional
        If provided, save the figure to this path

    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    rho = state.density_matrix()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot real part
    im1 = ax1.imshow(np.real(rho), cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
    ax1.set_title('Real Part of ρ')
    ax1.set_xlabel('Basis State')
    ax1.set_ylabel('Basis State')
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
    ax1.set_yticklabels(['⟨00|', '⟨01|', '⟨10|', '⟨11|'])
    plt.colorbar(im1, ax=ax1)

    # Add values as text
    for i in range(4):
        for j in range(4):
            text = ax1.text(j, i, f'{np.real(rho[i, j]):.2f}',
                           ha="center", va="center", color="black", fontsize=9)

    # Plot imaginary part
    im2 = ax2.imshow(np.imag(rho), cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
    ax2.set_title('Imaginary Part of ρ')
    ax2.set_xlabel('Basis State')
    ax2.set_ylabel('Basis State')
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'])
    ax2.set_yticklabels(['⟨00|', '⟨01|', '⟨10|', '⟨11|'])
    plt.colorbar(im2, ax=ax2)

    # Add values as text
    for i in range(4):
        for j in range(4):
            text = ax2.text(j, i, f'{np.imag(rho[i, j]):.2f}',
                           ha="center", va="center", color="black", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f'Density Matrix: {state.name if state.name else "Two-Qubit State"}',
                     fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


def plot_chsh_demonstration(
    results: dict = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive visualization of CHSH inequality violation.

    Parameters
    ----------
    results : dict, optional
        Results from demonstrate_bell_violation(). If None, will run the demo.
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    if results is None:
        results = demonstrate_bell_violation(shots=10000, num_trials=100)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Histogram of CHSH values
    ax1 = fig.add_subplot(gs[0, 0])
    n, bins, patches = ax1.hist(results['chsh_values'], bins=30, alpha=0.7,
                                 color='skyblue', edgecolor='black')

    # Color bars based on whether they violate classical bound
    for i, patch in enumerate(patches):
        if bins[i] > 2.0 or bins[i+1] < -2.0:
            patch.set_facecolor('green')
            patch.set_alpha(0.8)

    # Add vertical lines for bounds
    ax1.axvline(2.0, color='red', linestyle='--', linewidth=2, label='Classical Bound')
    ax1.axvline(-2.0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(quantum_bound(), color='blue', linestyle='--', linewidth=2, label='Quantum Limit')
    ax1.axvline(-quantum_bound(), color='blue', linestyle='--', linewidth=2)
    ax1.axvline(results['mean_chsh'], color='orange', linestyle='-', linewidth=2, label='Mean')

    ax1.set_xlabel('CHSH Parameter S', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of CHSH Values\n(Quantum Mechanics Violates Classical Bound)',
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Statistics summary
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    stats_text = f"""
    Bell's Inequality Violation Results
    {'='*40}

    Experimental Statistics:
    • Mean CHSH:        {results['mean_chsh']:.3f} ± {results['std_chsh']:.3f}
    • Exact Prediction: {results['exact_value']:.3f}

    Theoretical Bounds:
    • Classical Maximum:  {results['classical_bound']:.3f}
    • Quantum Maximum:    {results['quantum_bound']:.3f}

    Violation Analysis:
    • Violation Rate:     {results['violation_rate']*100:.1f}%
    • Excess over Classical: {results['mean_chsh'] - 2:.3f}
    • % Above Classical:  {(results['mean_chsh']/2 - 1)*100:.1f}%

    Conclusion:
    {'✓' if results['mean_chsh'] > 2 else '✗'} Quantum mechanics VIOLATES Bell's inequality!
    {'✓' if results['violation_rate'] > 0.95 else '✗'} Consistent violation observed.

    This proves quantum entanglement exhibits
    genuine non-local correlations that cannot
    be explained by any local hidden variable theory.
    """

    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Comparison bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    categories = ['Classical\nLimit', 'Quantum\nPrediction', 'Quantum\nLimit']
    values = [2.0, results['mean_chsh'], quantum_bound()]
    colors = ['red', 'green', 'blue']

    bars = ax3.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.axhline(2.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_ylabel('CHSH Parameter |S|', fontsize=12)
    ax3.set_title('Classical vs Quantum Predictions', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Angle scan
    ax4 = fig.add_subplot(gs[1, 1])
    scan_data = scan_chsh_angles(num_points=100)

    ax4.plot(scan_data['angles'] * 180/np.pi, scan_data['chsh_values'],
             linewidth=2, color='purple', label='CHSH(θ)')
    ax4.axhline(2.0, color='red', linestyle='--', linewidth=2, label='Classical Bound', alpha=0.7)
    ax4.axhline(-2.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.axhline(quantum_bound(), color='blue', linestyle='--', linewidth=1.5,
               label='Quantum Limit', alpha=0.7)
    ax4.axhline(-quantum_bound(), color='blue', linestyle='--', linewidth=1.5, alpha=0.7)

    # Shade violation region
    ax4.fill_between(scan_data['angles'] * 180/np.pi, 2.0, 3.0,
                     alpha=0.2, color='green', label='Violation Region')
    ax4.fill_between(scan_data['angles'] * 180/np.pi, -2.0, -3.0,
                     alpha=0.2, color='green')

    ax4.set_xlabel('Measurement Angle θ (degrees)', fontsize=12)
    ax4.set_ylabel('CHSH Parameter S', fontsize=12)
    ax4.set_title('CHSH Value vs Measurement Angle', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-3, 3])

    fig.suptitle('Bell\'s Inequality Violation: Quantum Non-Locality Demonstrated',
                 fontsize=18, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


def create_chsh_animation(
    num_frames: int = 100,
    save_path: Optional[str] = None,
    fps: int = 30
) -> animation.FuncAnimation:
    """
    Create an animated demonstration of CHSH violation building up.

    This is perfect for LinkedIn posts - shows the violation appearing
    as measurements accumulate.

    Parameters
    ----------
    num_frames : int
        Number of animation frames
    save_path : str, optional
        If provided, save as MP4 or GIF
    fps : int
        Frames per second for saved animation

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    state = bell_phi_plus()
    a, a_prime, b, b_prime = optimal_chsh_angles()

    # Data storage
    measurements_per_frame = 100
    all_chsh_values = []
    frame_numbers = []

    def init():
        ax1.clear()
        ax2.clear()
        return ax1, ax2

    def update(frame):
        # Run measurements for this frame
        S = compute_chsh_value(state, a, a_prime, b, b_prime,
                              shots=measurements_per_frame, exact=False)
        all_chsh_values.append(S)
        frame_numbers.append(frame)

        # Left plot: Histogram
        ax1.clear()
        ax1.hist(all_chsh_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(2.0, color='red', linestyle='--', linewidth=2, label='Classical Bound')
        ax1.axvline(quantum_bound(), color='blue', linestyle='--', linewidth=2, label='Quantum Limit')
        if all_chsh_values:
            mean_val = np.mean(all_chsh_values)
            ax1.axvline(mean_val, color='orange', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax1.set_xlabel('CHSH Parameter S')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'CHSH Values (n={len(all_chsh_values)} trials)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([1.5, 3.0])

        # Right plot: Running mean
        ax2.clear()
        if len(all_chsh_values) > 1:
            running_mean = np.cumsum(all_chsh_values) / np.arange(1, len(all_chsh_values) + 1)
            ax2.plot(frame_numbers, running_mean, 'g-', linewidth=2, label='Running Mean')

        ax2.axhline(2.0, color='red', linestyle='--', linewidth=2, label='Classical Bound', alpha=0.7)
        ax2.axhline(quantum_bound(), color='blue', linestyle='--', linewidth=2,
                   label='Quantum Limit', alpha=0.7)
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Mean CHSH Value')
        ax2.set_title('Convergence to Quantum Prediction')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([1.5, 3.0])

        fig.suptitle('Bell\'s Inequality Violation: Real-Time Demonstration',
                     fontsize=16, fontweight='bold')

        return ax1, ax2

    anim = animation.FuncAnimation(fig, update, frames=num_frames,
                                   init_func=init, blit=False, interval=50)

    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps, bitrate=1800)

    return anim


def plot_entanglement_comparison(save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare all four Bell states side by side.

    Shows density matrices and entanglement measures for educational purposes.

    Parameters
    ----------
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    from .bell_states import bell_phi_plus, bell_phi_minus, bell_psi_plus, bell_psi_minus

    states = [
        bell_phi_plus(),
        bell_phi_minus(),
        bell_psi_plus(),
        bell_psi_minus()
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    for col, state in enumerate(states):
        # Top row: Real part of density matrix
        ax = axes[0, col]
        rho = state.density_matrix()
        im = ax.imshow(np.real(rho), cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(f'|{state.name}⟩', fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xticklabels(['00', '01', '10', '11'], fontsize=9)
        ax.set_yticklabels(['00', '01', '10', '11'], fontsize=9)

        if col == 0:
            ax.set_ylabel('Real Part of ρ', fontsize=11)

        # Bottom row: Information
        ax = axes[1, col]
        ax.axis('off')

        # Calculate properties
        entropy = state.von_neumann_entropy()
        is_ent = state.is_entangled()
        schmidt_coeffs, _, _ = state.schmidt_decomposition()

        info_text = f"""
State: |{state.name}⟩

Properties:
• Entangled: {'Yes' if is_ent else 'No'}
• Entropy: {entropy:.4f}
• Max Entangled: {'Yes' if abs(entropy - 1.0) < 0.01 else 'No'}

Schmidt Coefficients:
• λ₁ = {schmidt_coeffs[0]:.4f}
• λ₂ = {schmidt_coeffs[1]:.4f}

Formula:
{state}
        """

        ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    fig.suptitle('Four Bell States: Maximally Entangled Two-Qubit Systems',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


def create_linkedin_summary(save_path: str = "plots/phase2/bell_violation_summary.png"):
    """
    Create a single, visually striking image perfect for LinkedIn.

    This combines the most important visualization elements into one
    compelling graphic.

    Parameters
    ----------
    save_path : str
        Where to save the image
    """
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    results = demonstrate_bell_violation(shots=10000, num_trials=200)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 1, 0.8])

    # Main title
    fig.suptitle('Bell\'s Inequality: Quantum Mechanics Violates Classical Physics',
                 fontsize=20, fontweight='bold', y=0.98)

    # Plot 1: The key result
    ax1 = fig.add_subplot(gs[0, :])
    categories = ['Classical\nTheory\nMaximum', 'Our Quantum\nExperiment', 'Quantum\nTheory\nMaximum']
    values = [2.0, results['mean_chsh'], quantum_bound()]
    colors = ['#d62728', '#2ca02c', '#1f77b4']

    bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
    ax1.axhline(2.0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_ylabel('CHSH Parameter S', fontsize=14, fontweight='bold')
    ax1.set_title('Quantum > Classical: Proof of Non-Locality', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylim([0, 3])
    ax1.grid(True, alpha=0.2, axis='y')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=13)

    # Annotation
    ax1.annotate('', xy=(1, results['mean_chsh']), xytext=(0, 2.0),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    violation_amount = results['mean_chsh'] - 2
    ax1.text(0.5, 2.4, f'Violation!\n+{violation_amount:.2f}',
            ha='center', fontsize=12, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Plot 2: Histogram
    ax2 = fig.add_subplot(gs[1, 0])
    n, bins, patches = ax2.hist(results['chsh_values'], bins=25, alpha=0.7,
                                color='skyblue', edgecolor='black')

    for i, patch in enumerate(patches):
        if bins[i] > 2.0:
            patch.set_facecolor('#2ca02c')
            patch.set_alpha(0.9)

    ax2.axvline(2.0, color='red', linestyle='--', linewidth=3, label='Classical Limit', alpha=0.8)
    ax2.axvline(results['mean_chsh'], color='orange', linestyle='-', linewidth=3,
               label=f'Mean: {results["mean_chsh"]:.3f}')
    ax2.set_xlabel('CHSH Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'200 Experimental Trials', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Info box
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    info_text = f"""
🔬 EXPERIMENTAL RESULTS

Measured Value: {results['mean_chsh']:.3f} ± {results['std_chsh']:.3f}

✓ {results['violation_rate']*100:.0f}% of trials violated classical bound
✓ {((results['mean_chsh']-2)/2*100):.1f}% above classical limit

📊 What This Means:

• Classical physics predicts S ≤ 2.000
• Quantum mechanics predicts S ≈ 2.828
• We measured S = {results['mean_chsh']:.3f}

➜ This PROVES quantum entanglement
  exhibits non-local correlations!

➜ No "hidden variable" theory can
  explain these results.

➜ Einstein's "spooky action at a
  distance" is REAL! 👻
    """

    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
            fontsize=11.5, verticalalignment='top', fontfamily='sans-serif',
            bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0',
                     edgecolor='black', linewidth=2))

    # Plot 4: Visual explanation
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    explanation = """
    WHY THIS MATTERS: Bell's inequality proves that quantum mechanics is fundamentally different from classical physics.
    Entangled particles exhibit correlations that CANNOT be explained by any local theory where particles have definite
    properties before measurement. This experiment demonstrates the "spooky action at a distance" that troubled Einstein.
    """

    ax4.text(0.5, 0.5, explanation, transform=ax4.transAxes,
            ha='center', va='center', fontsize=12, style='italic',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='yellow', alpha=0.3, edgecolor='orange', linewidth=2))

    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"✓ LinkedIn summary saved to: {save_path}")

    return fig
