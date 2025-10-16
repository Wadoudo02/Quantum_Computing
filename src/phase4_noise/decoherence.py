# -*- coding: utf-8 -*-
"""
Time-Dependent Decoherence Simulation

Simulates T1 (energy relaxation) and T2 (dephasing) decoherence processes
that affect real quantum systems over time.

Based on Imperial College Notes - Section 4.2: Open quantum systems and decoherence

Key Concepts:
- T1: Energy relaxation time (|1> -> |0>)
- T2: Dephasing time (loss of coherence)
- T2*: Inhomogeneous dephasing
- Physical constraint: T2 <= 2*T1

Typical Hardware Values:
- Superconducting qubits: T1 ~ 50-100 us, T2 ~ 20-80 us
- Ion traps: T1 > 10 s, T2 ~ 1 s
- NV centers: T1 ~ ms, T2 ~ ms

Author: Wadoud Charbak
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from .density_matrix import DensityMatrix, pure_state_density_matrix, fidelity
from .quantum_channels import amplitude_damping_channel, phase_damping_channel


class DecoherenceSimulator:
    """
    Simulates decoherence processes over time.

    Attributes
    ----------
    T1 : float
        Energy relaxation time (amplitude damping)
    T2 : float
        Dephasing time (phase damping)
    initial_state : DensityMatrix
        Initial quantum state

    Examples
    --------
    >>> sim = DecoherenceSimulator(T1=100e-6, T2=50e-6)
    >>> times, states = sim.simulate([0, 10e-6, 20e-6, 50e-6])
    >>> # Analyze fidelity decay
    """

    def __init__(self, T1: float, T2: float, initial_state: Optional[DensityMatrix] = None):
        """
        Initialize decoherence simulator.

        Parameters
        ----------
        T1 : float
            Energy relaxation time (seconds)
        T2 : float
            Dephasing time (seconds)
        initial_state : DensityMatrix, optional
            Initial state (default: |0>)

        Raises
        ------
        ValueError
            If T2 > 2*T1 (violates physical constraint)
        """
        if T2 > 2 * T1:
            raise ValueError(f"T2 ({T2}) cannot exceed 2*T1 ({2*T1})")

        self.T1 = T1
        self.T2 = T2

        if initial_state is None:
            self.initial_state = pure_state_density_matrix([1, 0])  # |0>
        else:
            self.initial_state = initial_state

    def simulate(self, time_points: np.ndarray) -> Tuple[np.ndarray, List[DensityMatrix]]:
        """
        Simulate decoherence over specified time points.

        Parameters
        ----------
        time_points : np.ndarray
            Array of time values (seconds)

        Returns
        -------
        times : np.ndarray
            Time points
        states : List[DensityMatrix]
            Density matrices at each time point

        Examples
        --------
        >>> sim = DecoherenceSimulator(T1=100, T2=50)
        >>> times, states = sim.simulate(np.linspace(0, 200, 50))
        """
        states = []

        for t in time_points:
            # Compute decay parameters
            gamma = 1 - np.exp(-t / self.T1)  # Amplitude damping
            lambda_ = 1 - np.exp(-t / self.T2)  # Phase damping

            # Apply channels
            rho_t = amplitude_damping_channel(self.initial_state, gamma)
            rho_t = phase_damping_channel(rho_t, lambda_)

            states.append(rho_t)

        return time_points, states

    def compute_fidelity_decay(self, time_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute fidelity vs time.

        Parameters
        ----------
        time_points : np.ndarray
            Array of time values

        Returns
        -------
        times : np.ndarray
            Time points
        fidelities : np.ndarray
            Fidelity at each time point
        """
        times, states = self.simulate(time_points)
        fidelities = np.array([fidelity(self.initial_state, rho) for rho in states])
        return times, fidelities

    def compute_purity_decay(self, time_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute purity vs time.

        Parameters
        ----------
        time_points : np.ndarray
            Array of time values

        Returns
        -------
        times : np.ndarray
            Time points
        purities : np.ndarray
            Purity at each time point
        """
        times, states = self.simulate(time_points)
        purities = np.array([rho.purity() for rho in states])
        return times, purities

    def compute_population_evolution(self, time_points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Track population (diagonal elements) evolution.

        Parameters
        ----------
        time_points : np.ndarray
            Array of time values

        Returns
        -------
        dict
            Dictionary with 'times', 'pop_0' (ground), 'pop_1' (excited)
        """
        times, states = self.simulate(time_points)

        pop_0 = np.array([rho.matrix[0, 0].real for rho in states])
        pop_1 = np.array([rho.matrix[1, 1].real for rho in states])

        return {
            'times': times,
            'pop_0': pop_0,
            'pop_1': pop_1
        }

    def compute_coherence_evolution(self, time_points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Track coherence (off-diagonal elements) evolution.

        Parameters
        ----------
        time_points : np.ndarray
            Array of time values

        Returns
        -------
        dict
            Dictionary with 'times', 'coherence' (|rho_01|)
        """
        times, states = self.simulate(time_points)

        coherence = np.array([np.abs(rho.matrix[0, 1]) for rho in states])

        return {
            'times': times,
            'coherence': coherence
        }


def simulate_t1_decay(initial_state: DensityMatrix, T1: float,
                      time_points: np.ndarray) -> Tuple[np.ndarray, List[DensityMatrix]]:
    """
    Simulate pure T1 (amplitude damping) decay.

    Models |1> -> |0> energy relaxation without dephasing.

    Parameters
    ----------
    initial_state : DensityMatrix
        Initial quantum state
    T1 : float
        Energy relaxation time
    time_points : np.ndarray
        Time points for simulation

    Returns
    -------
    times : np.ndarray
        Time array
    states : List[DensityMatrix]
        States at each time point

    Examples
    --------
    >>> # Excited state decaying
    >>> rho_1 = pure_state_density_matrix([0, 1])
    >>> times, states = simulate_t1_decay(rho_1, T1=100, time_points=np.linspace(0, 300, 50))
    """
    states = []

    for t in time_points:
        gamma = 1 - np.exp(-t / T1)
        rho_t = amplitude_damping_channel(initial_state, gamma)
        states.append(rho_t)

    return time_points, states


def simulate_t2_decay(initial_state: DensityMatrix, T2: float,
                      time_points: np.ndarray) -> Tuple[np.ndarray, List[DensityMatrix]]:
    """
    Simulate pure T2 (phase damping) decay.

    Models loss of coherence without energy relaxation.

    Parameters
    ----------
    initial_state : DensityMatrix
        Initial quantum state
    T2 : float
        Dephasing time
    time_points : np.ndarray
        Time points for simulation

    Returns
    -------
    times : np.ndarray
        Time array
    states : List[DensityMatrix]
        States at each time point

    Examples
    --------
    >>> # Superposition losing coherence
    >>> rho_plus = pure_state_density_matrix([1, 1])
    >>> times, states = simulate_t2_decay(rho_plus, T2=50, time_points=np.linspace(0, 200, 50))
    """
    states = []

    for t in time_points:
        lambda_ = 1 - np.exp(-t / T2)
        rho_t = phase_damping_channel(initial_state, lambda_)
        states.append(rho_t)

    return time_points, states


def simulate_combined_decay(initial_state: DensityMatrix, T1: float, T2: float,
                            time_points: np.ndarray) -> Tuple[np.ndarray, List[DensityMatrix]]:
    """
    Simulate combined T1 + T2 decoherence.

    Realistic model including both energy relaxation and dephasing.

    Parameters
    ----------
    initial_state : DensityMatrix
        Initial quantum state
    T1 : float
        Energy relaxation time
    T2 : float
        Dephasing time (must satisfy T2 <= 2*T1)
    time_points : np.ndarray
        Time points for simulation

    Returns
    -------
    times : np.ndarray
        Time array
    states : List[DensityMatrix]
        States at each time point

    Examples
    --------
    >>> # Realistic superconducting qubit decoherence
    >>> rho = pure_state_density_matrix([1, 1])
    >>> times, states = simulate_combined_decay(rho, T1=100e-6, T2=50e-6,
    ...                                         time_points=np.linspace(0, 200e-6, 100))
    """
    if T2 > 2 * T1:
        raise ValueError(f"T2 ({T2}) cannot exceed 2*T1 ({2*T1})")

    sim = DecoherenceSimulator(T1, T2, initial_state)
    return sim.simulate(time_points)


def ramsey_experiment(initial_state: DensityMatrix, T2: float, omega: float,
                      time_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Ramsey interferometry experiment.

    Measures T2 dephasing through oscillating measurement probability.

    Sequence: H - free evolution (t) - H - measure
    Result: P(0) = (1 + exp(-t/T2) * cos(omega*t)) / 2

    Parameters
    ----------
    initial_state : DensityMatrix
        Initial state (typically |0>)
    T2 : float
        Dephasing time
    omega : float
        Detuning frequency (rad/s)
    time_points : np.ndarray
        Free evolution times

    Returns
    -------
    times : np.ndarray
        Time array
    probabilities : np.ndarray
        P(0) at each time point

    Examples
    --------
    >>> # Measure T2 = 50 us with 1 MHz detuning
    >>> rho_0 = pure_state_density_matrix([1, 0])
    >>> times, probs = ramsey_experiment(rho_0, T2=50e-6, omega=2*np.pi*1e6,
    ...                                  time_points=np.linspace(0, 100e-6, 100))
    """
    # Hadamard gate
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    probabilities = []

    for t in time_points:
        # Apply first Hadamard
        rho = initial_state.apply_unitary(H)

        # Free evolution with dephasing
        # Rotation + dephasing
        U_rot = np.array([[1, 0], [0, np.exp(-1j * omega * t)]])
        rho = rho.apply_unitary(U_rot)

        lambda_ = 1 - np.exp(-t / T2)
        rho = phase_damping_channel(rho, lambda_)

        # Apply second Hadamard
        rho = rho.apply_unitary(H)

        # Measure |0> probability
        prob_0 = rho.matrix[0, 0].real
        probabilities.append(prob_0)

    return time_points, np.array(probabilities)


def extract_t1_from_decay(times: np.ndarray, populations: np.ndarray) -> float:
    """
    Extract T1 from exponential decay fit.

    Fits P_1(t) = P_1(0) * exp(-t/T1)

    Parameters
    ----------
    times : np.ndarray
        Time points
    populations : np.ndarray
        Excited state population at each time

    Returns
    -------
    float
        Estimated T1
    """
    # Fit exponential decay
    # log(P) = log(P0) - t/T1
    log_pop = np.log(populations + 1e-15)  # Avoid log(0)
    coeffs = np.polyfit(times, log_pop, 1)
    T1_est = -1 / coeffs[0]
    return T1_est


def extract_t2_from_ramsey(times: np.ndarray, probabilities: np.ndarray) -> float:
    """
    Extract T2 from Ramsey oscillations.

    Fits envelope decay of oscillating signal.

    Parameters
    ----------
    times : np.ndarray
        Time points
    probabilities : np.ndarray
        P(0) measurements

    Returns
    -------
    float
        Estimated T2
    """
    # Extract envelope (simplified - assumes known frequency)
    # Full implementation would fit: A * exp(-t/T2) * cos(omega*t + phi) + offset
    from scipy.signal import hilbert

    # Compute envelope using Hilbert transform
    envelope = np.abs(hilbert(probabilities - 0.5))

    # Fit exponential to envelope
    log_env = np.log(envelope + 1e-15)
    coeffs = np.polyfit(times, log_env, 1)
    T2_est = -1 / coeffs[0]

    return T2_est


# Self-test
if __name__ == "__main__":
    print("=" * 70)
    print("DECOHERENCE SIMULATION MODULE - SELF TEST")
    print("=" * 70)

    # Test 1: T1 decay of excited state
    print("\n1. T1 Decay: |1> -> |0>")
    rho_1 = pure_state_density_matrix([0, 1])
    T1 = 100  # arbitrary units
    times = np.linspace(0, 300, 50)
    times_t1, states_t1 = simulate_t1_decay(rho_1, T1, times)

    print(f"   Initial population in |1>: {rho_1.matrix[1,1].real:.4f}")
    print(f"   After t=T1: {states_t1[len(states_t1)//3].matrix[1,1].real:.4f}")
    print(f"   After t=3*T1: {states_t1[-1].matrix[1,1].real:.4f}")
    print(f"   Expected: ~0.368 at T1, ~0.05 at 3*T1")

    # Test 2: T2 dephasing of superposition
    print("\n2. T2 Dephasing: |+> losing coherence")
    rho_plus = pure_state_density_matrix([1, 1])
    T2 = 50
    times_t2, states_t2 = simulate_t2_decay(rho_plus, T2, np.linspace(0, 150, 50))

    initial_coherence = np.abs(rho_plus.matrix[0,1])
    final_coherence = np.abs(states_t2[-1].matrix[0,1])
    print(f"   Initial coherence: {initial_coherence:.4f}")
    print(f"   After t=3*T2: {final_coherence:.4f}")
    print(f"   Coherence reduced to: {final_coherence/initial_coherence*100:.1f}%")

    # Test 3: Combined T1 + T2
    print("\n3. Combined T1 + T2 Decoherence")
    sim = DecoherenceSimulator(T1=100, T2=50, initial_state=rho_plus)
    times_comb = np.linspace(0, 200, 50)
    times_out, states_comb = sim.simulate(times_comb)

    print(f"   Initial purity: {sim.initial_state.purity():.4f}")
    print(f"   After t=50: {states_comb[len(states_comb)//4].purity():.4f}")
    print(f"   After t=200: {states_comb[-1].purity():.4f}")

    # Test 4: Fidelity decay
    print("\n4. Fidelity Decay")
    times_fid, fidelities = sim.compute_fidelity_decay(times_comb)
    print(f"   Initial fidelity: {fidelities[0]:.4f}")
    print(f"   At t=50: {fidelities[len(fidelities)//4]:.4f}")
    print(f"   At t=200: {fidelities[-1]:.4f}")

    # Test 5: Population evolution
    print("\n5. Population Evolution (|1> decaying)")
    rho_1 = pure_state_density_matrix([0, 1])
    sim_pop = DecoherenceSimulator(T1=100, T2=50, initial_state=rho_1)
    pop_data = sim_pop.compute_population_evolution(np.linspace(0, 300, 50))
    print(f"   Initial P(|1>): {pop_data['pop_1'][0]:.4f}")
    print(f"   Final P(|1>): {pop_data['pop_1'][-1]:.4f}")
    print(f"   Final P(|0>): {pop_data['pop_0'][-1]:.4f}")
    print(f"   Energy transferred to environment!")

    # Test 6: Coherence decay
    print("\n6. Coherence Decay (|+> state)")
    rho_plus = pure_state_density_matrix([1, 1])
    sim_coh = DecoherenceSimulator(T1=100, T2=50, initial_state=rho_plus)
    coh_data = sim_coh.compute_coherence_evolution(np.linspace(0, 150, 50))
    print(f"   Initial coherence: {coh_data['coherence'][0]:.4f}")
    print(f"   After t=150: {coh_data['coherence'][-1]:.4f}")
    print(f"   Decay rate controlled by T2")

    # Test 7: Ramsey experiment
    print("\n7. Ramsey Experiment")
    rho_0 = pure_state_density_matrix([1, 0])
    omega = 2 * np.pi * 0.1  # 0.1 Hz in arbitrary units
    times_ramsey = np.linspace(0, 100, 100)
    times_r, probs_r = ramsey_experiment(rho_0, T2=50, omega=omega, time_points=times_ramsey)
    print(f"   Initial P(0): {probs_r[0]:.4f}")
    print(f"   Shows oscillations with decay envelope")
    print(f"   Oscillation frequency: {omega/(2*np.pi):.3f} Hz")

    print("\n" + "=" * 70)
    print("ALL DECOHERENCE TESTS COMPLETED!")
    print("=" * 70)
