#!/usr/bin/env python3
"""
Phase 2 Quick Demo: Bell's Inequality Violation
================================================

This script demonstrates the key features of Phase 2:
- Creating Bell states
- Testing CHSH inequality
- Generating visualizations

Run from project root: python examples/phase2_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from phase2_entanglement.bell_states import bell_phi_plus, bell_phi_minus, bell_psi_plus, bell_psi_minus
from phase2_entanglement.bells_inequality import demonstrate_bell_violation, classical_bound, quantum_bound
from phase2_entanglement.visualization import plot_chsh_demonstration, create_linkedin_summary
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def main():
    print("=" * 70)
    print("Phase 2: Demonstrating Quantum Non-Locality")
    print("=" * 70)

    # Part 1: Bell States
    print("\n📊 Part 1: Creating and Analyzing Bell States")
    print("-" * 70)

    states = [
        ('Φ+', bell_phi_plus()),
        ('Φ-', bell_phi_minus()),
        ('Ψ+', bell_psi_plus()),
        ('Ψ-', bell_psi_minus())
    ]

    for name, state in states:
        entropy = state.von_neumann_entropy()
        coeffs, _, _ = state.schmidt_decomposition()
        print(f"|{name}⟩:")
        print(f"  Maximally entangled: {entropy > 0.99}")
        print(f"  von Neumann entropy: {entropy:.4f} bits")
        print(f"  Schmidt coefficients: [{coeffs[0]:.4f}, {coeffs[1]:.4f}]")
        print()

    # Part 2: CHSH Inequality Test
    print("\n🔔 Part 2: Testing Bell's Inequality (CHSH)")
    print("-" * 70)
    print(f"Classical bound (local realism): |S| ≤ {classical_bound()}")
    print(f"Quantum bound (Tsirelson): |S| ≤ {quantum_bound():.4f}")
    print()

    print("Running 100 trials with 10,000 measurements each...")
    results = demonstrate_bell_violation(shots=10000, num_trials=100)

    print(f"\n✨ Results:")
    print(f"  Mean CHSH value: {results['mean_chsh']:.3f} ± {results['std_chsh']:.3f}")
    print(f"  Exact prediction: {results['exact_value']:.4f}")
    print(f"  Violation rate: {results['violation_rate']*100:.1f}%")
    print(f"  Exceeds classical by: +{(results['mean_chsh']/2.0 - 1)*100:.1f}%")

    if results['mean_chsh'] > classical_bound():
        print("\n🎉 VIOLATION CONFIRMED! Quantum mechanics exhibits non-local correlations!")

    # Part 3: Visualizations
    print("\n📈 Part 3: Generating Visualizations")
    print("-" * 70)

    plots_dir = Path(__file__).parent.parent / 'plots' / 'phase2'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Creating comprehensive CHSH demonstration plot...")
    plot_chsh_demonstration(results, save_path=str(plots_dir / 'chsh_demo.png'))
    print(f"  ✓ Saved to {plots_dir / 'chsh_demo.png'}")

    print("Creating LinkedIn-ready summary graphic...")
    create_linkedin_summary(save_path=str(plots_dir / 'linkedin_summary.png'))
    print(f"  ✓ Saved to {plots_dir / 'linkedin_summary.png'}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ Phase 2 Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • All four Bell states are maximally entangled (entropy = 1 bit)")
    print("  • CHSH test consistently violates classical bound")
    print("  • Quantum mechanics achieves S ≈ 2.83 vs classical limit of 2.0")
    print("  • This proves genuine quantum non-locality!")
    print("\nNext steps:")
    print("  • Run the Streamlit app: streamlit run src/phase2_entanglement/app.py")
    print("  • Explore the interactive demonstrations")
    print("  • Generate animations for social media")
    print()


if __name__ == '__main__':
    main()
