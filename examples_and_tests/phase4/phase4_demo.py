#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 Demo: Quantum Noise and Decoherence

Comprehensive demonstration of:
1. Density matrix formalism
2. Six quantum noise channels
3. T1/T2 decoherence simulation
4. Fidelity and purity decay
5. Why quantum computers are so hard to build

Author: Wadoud Charbak
For: Quantinuum & Riverlane recruitment
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.phase4_noise import (
    # Density matrices
    DensityMatrix,
    pure_state_density_matrix,
    mixed_state_density_matrix,
    fidelity,
    purity,
    # Noise channels
    bit_flip_channel,
    phase_flip_channel,
    depolarizing_channel,
    amplitude_damping_channel,
    phase_damping_channel,
    # Decoherence
    DecoherenceSimulator,
    simulate_t1_decay,
    simulate_t2_decay,
    simulate_combined_decay,
    ramsey_experiment,
)


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_density_matrices():
    """Demonstrate density matrix formalism."""
    print_section("1. DENSITY MATRIX FORMALISM")

    print("\n1.1 Pure State |0>")
    rho_0 = pure_state_density_matrix([1, 0])
    print(rho_0)
    print(f"\nMatrix:\n{rho_0.matrix}")

    print("\n1.2 Pure State |+> = (|0> + |1>)/sqrt(2)")
    rho_plus = pure_state_density_matrix([1, 1])
    print(rho_plus)
    x, y, z = rho_plus.bloch_vector()
    print(f"Bloch vector: ({x:.4f}, {y:.4f}, {z:.4f})")

    print("\n1.3 Maximally Mixed State (50% |0>, 50% |1>)")
    rho_mixed = DensityMatrix.maximally_mixed(2)
    print(rho_mixed)
    print(f"\nMatrix:\n{rho_mixed.matrix}")
    print(f"Bloch vector: {rho_mixed.bloch_vector()} (at center)")

    print("\n1.4 Fidelity Between States")
    rho_1 = pure_state_density_matrix([0, 1])
    print(f"F(|0>, |0>) = {fidelity(rho_0, rho_0):.6f}")
    print(f"F(|0>, |1>) = {fidelity(rho_0, rho_1):.6f}")
    print(f"F(|0>, |+>) = {fidelity(rho_0, rho_plus):.6f}")
    print(f"F(|+>, mixed) = {fidelity(rho_plus, rho_mixed):.6f}")


def demo_noise_channels():
    """Demonstrate quantum noise channels."""
    print_section("2. QUANTUM NOISE CHANNELS")

    print("\n2.1 Bit-Flip Channel (10% error rate)")
    rho = pure_state_density_matrix([1, 0])
    print(f"Initial: |0> state, purity = {rho.purity():.4f}")
    rho_bf = bit_flip_channel(rho, 0.1)
    print(f"After bit-flip: purity = {rho_bf.purity():.4f}")
    print(f"Population: 90% |0>, 10% |1>")
    print(f"<0|rho|0> = {rho_bf.matrix[0,0].real:.4f}")
    print(f"<1|rho|1> = {rho_bf.matrix[1,1].real:.4f}")

    print("\n2.2 Phase-Flip Channel (20% error rate)")
    rho = pure_state_density_matrix([1, 1])
    x_init, _, _ = rho.bloch_vector()
    print(f"Initial: |+> state, Bloch x = {x_init:.4f}")
    rho_pf = phase_flip_channel(rho, 0.2)
    x_final, _, _ = rho_pf.bloch_vector()
    print(f"After phase-flip: Bloch x = {x_final:.4f}")
    print(f"Coherence reduced: {x_final/x_init*100:.1f}%")

    print("\n2.3 Depolarizing Channel (30% error rate)")
    rho = pure_state_density_matrix([1, 0])
    print(f"Initial purity: {rho.purity():.4f}")
    rho_depol = depolarizing_channel(rho, 0.3)
    print(f"After depolarizing: {rho_depol.purity():.4f}")
    print(f"State becoming mixed!")

    print("\n2.4 Amplitude Damping (T1 - Energy Loss)")
    rho = pure_state_density_matrix([0, 1])  # |1>
    print(f"Initial: |1> (excited state)")
    print(f"Population in |1>: {rho.matrix[1,1].real:.4f}")
    rho_ad = amplitude_damping_channel(rho, 0.5)
    print(f"After gamma=0.5 damping:")
    print(f"Population in |0>: {rho_ad.matrix[0,0].real:.4f}")
    print(f"Population in |1>: {rho_ad.matrix[1,1].real:.4f}")
    print("Energy transferred to environment (e.g., photon emission)!")

    print("\n2.5 Phase Damping (T2 - Dephasing)")
    rho = pure_state_density_matrix([1, 1])
    coherence_init = np.abs(rho.matrix[0,1])
    print(f"Initial: |+> state")
    print(f"Coherence: {coherence_init:.4f}")
    rho_pd = phase_damping_channel(rho, 0.5)
    coherence_final = np.abs(rho_pd.matrix[0,1])
    print(f"After lambda=0.5 dephasing:")
    print(f"Coherence: {coherence_final:.4f}")
    print(f"Coherence lost: {(1 - coherence_final/coherence_init)*100:.1f}%")
    print("Populations unchanged (no energy loss)")


def demo_t1_t2_decoherence():
    """Demonstrate T1/T2 decoherence."""
    print_section("3. T1 AND T2 DECOHERENCE")

    print("\n3.1 T1 Decay: Excited State |1> -> Ground State |0>")
    rho_1 = pure_state_density_matrix([0, 1])
    T1 = 100  # arbitrary time units
    times = np.linspace(0, 300, 50)

    times_t1, states_t1 = simulate_t1_decay(rho_1, T1, times)

    print(f"T1 = {T1}")
    print(f"Initial population in |1>: {rho_1.matrix[1,1].real:.4f}")
    idx_t1 = len(states_t1) // 3
    print(f"After t=T1: {states_t1[idx_t1].matrix[1,1].real:.4f} (expected: ~0.368)")
    print(f"After t=3*T1: {states_t1[-1].matrix[1,1].real:.4f} (expected: ~0.05)")

    print("\n3.2 T2 Dephasing: Superposition |+> Losing Coherence")
    rho_plus = pure_state_density_matrix([1, 1])
    T2 = 50
    times = np.linspace(0, 150, 50)

    times_t2, states_t2 = simulate_t2_decay(rho_plus, T2, times)

    initial_coh = np.abs(rho_plus.matrix[0,1])
    final_coh = np.abs(states_t2[-1].matrix[0,1])
    print(f"T2 = {T2}")
    print(f"Initial coherence: {initial_coh:.4f}")
    print(f"After t=3*T2: {final_coh:.4f}")
    print(f"Coherence remaining: {final_coh/initial_coh*100:.1f}%")

    print("\n3.3 Combined T1 + T2 (Realistic Decoherence)")
    sim = DecoherenceSimulator(T1=100, T2=50, initial_state=rho_plus)
    times = np.linspace(0, 200, 50)
    _, states = sim.simulate(times)

    print(f"T1 = {sim.T1}, T2 = {sim.T2}")
    print(f"Physical constraint: T2 <= 2*T1 satisfied!")
    print(f"Initial purity: {sim.initial_state.purity():.4f}")
    print(f"After t=200: {states[-1].purity():.4f}")

    print("\n3.4 Fidelity Decay")
    times_fid, fidelities = sim.compute_fidelity_decay(times)
    print(f"Initial fidelity: {fidelities[0]:.4f}")
    print(f"After t=50: {fidelities[len(fidelities)//4]:.4f}")
    print(f"After t=200: {fidelities[-1]:.4f}")
    print("State becoming increasingly different from initial!")


def demo_real_hardware_parameters():
    """Show realistic hardware decoherence times."""
    print_section("4. REAL QUANTUM HARDWARE PARAMETERS")

    print("\n4.1 Superconducting Qubits (e.g., IBM, Google)")
    T1_sc = 100e-6  # 100 microseconds
    T2_sc = 50e-6   # 50 microseconds
    print(f"T1 ~ {T1_sc*1e6:.0f} us")
    print(f"T2 ~ {T2_sc*1e6:.0f} us")
    print("Operating temperature: ~10-20 mK (millikelvin!)")
    print("Why so cold? Minimize thermal excitations")

    print("\n4.2 Ion Trap Qubits (e.g., IonQ)")
    T1_ion = 10  # seconds
    T2_ion = 1   # second
    print(f"T1 > {T1_ion} s")
    print(f"T2 ~ {T2_ion} s")
    print("Much better coherence! But slower gates (~10-100 us)")

    print("\n4.3 NV Centers (Diamond)")
    T1_nv = 1e-3  # milliseconds
    T2_nv = 1e-3  # milliseconds
    print(f"T1 ~ {T1_nv*1e3:.0f} ms")
    print(f"T2 ~ {T2_nv*1e3:.0f} ms")
    print("Room temperature operation possible!")

    print("\n4.4 Gate Times vs Coherence Times")
    gate_time_sc = 20e-9  # 20 nanoseconds
    gate_time_ion = 50e-6  # 50 microseconds

    print(f"\nSuperconducting qubits:")
    print(f"  Gate time: {gate_time_sc*1e9:.0f} ns")
    print(f"  T2 time: {T2_sc*1e6:.0f} us")
    print(f"  Gates before decoherence: ~{T2_sc/gate_time_sc:.0f}")

    print(f"\nIon traps:")
    print(f"  Gate time: {gate_time_ion*1e6:.0f} us")
    print(f"  T2 time: {T2_ion} s")
    print(f"  Gates before decoherence: ~{T2_ion/gate_time_ion:.0f}")


def demo_why_quantum_is_hard():
    """Explain why quantum computers are so challenging."""
    print_section("5. WHY QUANTUM COMPUTERS ARE SO HARD")

    print("\n5.1 Decoherence Kills Quantum Advantage")
    print("- Algorithms need coherent superposition")
    print("- Decoherence destroys superposition")
    print("- T2 times: microseconds to seconds")
    print("- Complex algorithms: thousands of gates")
    print("=> Need error correction! (Phase 5)")

    print("\n5.2 Error Rates")
    print("Modern hardware:")
    print("- Single-qubit gate error: ~0.1%")
    print("- Two-qubit gate error: ~0.5-1%")
    print("- Measurement error: ~1-2%")
    print("\nFor fault-tolerant quantum computing:")
    print("- Need: < 0.01% error rates")
    print("- Or: error correction overhead (100x-1000x qubits!)")

    print("\n5.3 Environmental Coupling")
    print("Qubits couple to:")
    print("- Thermal photons (requires cooling)")
    print("- Magnetic field fluctuations")
    print("- Electric field noise")
    print("- Phonons (vibrations)")
    print("- Cosmic rays(!)")
    print("=> Extreme isolation needed")

    print("\n5.4 The NISQ Era")
    print("NISQ = Noisy Intermediate-Scale Quantum")
    print("- 50-1000 qubits")
    print("- No error correction")
    print("- Limited circuit depth")
    print("- Exploring quantum advantage")
    print("=> Phase 5 will implement error correction!")


def main():
    """Run full Phase 4 demonstration."""
    print("\n" + "=" * 70)
    print("  PHASE 4: QUANTUM NOISE & DECOHERENCE")
    print("  Comprehensive Demonstration")
    print("=" * 70)
    print("\nWhy quantum computers are so hard to build...")
    print("and why we need quantum error correction!")

    demo_density_matrices()
    demo_noise_channels()
    demo_t1_t2_decoherence()
    demo_real_hardware_parameters()
    demo_why_quantum_is_hard()

    print("\n" + "=" * 70)
    print("  PHASE 4 DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Density matrices describe mixed quantum states")
    print("2. Six fundamental noise channels (Kraus operators)")
    print("3. T1 (energy relaxation) and T2 (dephasing)")
    print("4. Real hardware: T1 ~ 50-100 us (superconducting)")
    print("5. Decoherence destroys quantum advantage")
    print("6. Error correction is mandatory (Phase 5!)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
