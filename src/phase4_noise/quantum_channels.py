# -*- coding: utf-8 -*-
"""
Quantum Noise Channels using Kraus Operators

Implements the six fundamental quantum noise channels that describe
how real quantum systems interact with their environment and lose coherence.

Based on Imperial College Notes - Section 4.2: Open quantum systems and decoherence

Key Concepts:
- Quantum channel: rho -> sum_i K_i rho K_i_dagger
- Kraus operators satisfy: sum_i K_i_dagger K_i = I (completeness)
- CPTP maps (Completely Positive, Trace Preserving)

Channels Implemented:
1. Bit-flip (sigma_x error)
2. Phase-flip (sigma_z error)
3. Bit-phase-flip (sigma_y error)
4. Depolarizing (all Pauli errors)
5. Amplitude damping (T1 - energy relaxation)
6. Phase damping (T2 - pure dephasing)

Author: Wadoud Charbak
"""

import numpy as np
from typing import List, Tuple, Union
from .density_matrix import DensityMatrix


def verify_kraus_completeness(kraus_operators: List[np.ndarray], tolerance: float = 1e-10) -> bool:
    """
    Verify that Kraus operators satisfy completeness relation.

    sum_i K_i_dagger K_i = I

    Parameters
    ----------
    kraus_operators : List[np.ndarray]
        List of Kraus operators
    tolerance : float
        Numerical tolerance

    Returns
    -------
    bool
        True if completeness relation satisfied

    Examples
    --------
    >>> # Identity channel
    >>> K = [np.eye(2)]
    >>> verify_kraus_completeness(K)
    True
    """
    dim = kraus_operators[0].shape[0]
    sum_K = sum(K.conj().T @ K for K in kraus_operators)
    identity = np.eye(dim)
    return np.allclose(sum_K, identity, atol=tolerance)


def apply_channel(rho: Union[DensityMatrix, np.ndarray],
                  kraus_operators: List[np.ndarray]) -> DensityMatrix:
    """
    Apply quantum channel to density matrix.

    rho_out = sum_i K_i rho K_i_dagger

    Parameters
    ----------
    rho : DensityMatrix or np.ndarray
        Input density matrix
    kraus_operators : List[np.ndarray]
        Kraus operators defining the channel

    Returns
    -------
    DensityMatrix
        Output density matrix after noise

    Examples
    --------
    >>> rho = pure_state_density_matrix([1, 0])
    >>> K_bit_flip = [np.sqrt(0.9) * np.eye(2), np.sqrt(0.1) * sigma_x]
    >>> rho_noisy = apply_channel(rho, K_bit_flip)
    """
    # Extract matrix
    if isinstance(rho, DensityMatrix):
        matrix = rho.matrix
    else:
        matrix = rho

    # Apply channel: sum_i K_i rho K_i_dagger
    result = sum(K @ matrix @ K.conj().T for K in kraus_operators)

    return DensityMatrix(result)


# Pauli matrices (used in multiple channels)
_PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def bit_flip_channel(rho: Union[DensityMatrix, np.ndarray], p: float) -> DensityMatrix:
    """
    Bit-flip channel: applies sigma_x with probability p.

    Physical meaning: qubit randomly flips |0> <-> |1>

    Kraus operators:
    - K0 = sqrt(1-p) * I
    - K1 = sqrt(p) * sigma_x

    Parameters
    ----------
    rho : DensityMatrix or np.ndarray
        Input state
    p : float
        Bit-flip probability (0 <= p <= 1)

    Returns
    -------
    DensityMatrix
        State after bit-flip noise

    Examples
    --------
    >>> # Pure state |0> with 10% bit-flip noise
    >>> rho = pure_state_density_matrix([1, 0])
    >>> rho_noisy = bit_flip_channel(rho, 0.1)
    >>> # Now 90% |0>, 10% |1>
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Probability p must be in [0, 1], got {p}")

    kraus_ops = [
        np.sqrt(1 - p) * _PAULI_I,  # No error
        np.sqrt(p) * _PAULI_X        # Bit flip
    ]

    return apply_channel(rho, kraus_ops)


def phase_flip_channel(rho: Union[DensityMatrix, np.ndarray], p: float) -> DensityMatrix:
    """
    Phase-flip channel: applies sigma_z with probability p.

    Physical meaning: relative phase randomly flips |+> <-> |->

    Kraus operators:
    - K0 = sqrt(1-p) * I
    - K1 = sqrt(p) * sigma_z

    Parameters
    ----------
    rho : DensityMatrix or np.ndarray
        Input state
    p : float
        Phase-flip probability (0 <= p <= 1)

    Returns
    -------
    DensityMatrix
        State after phase-flip noise

    Examples
    --------
    >>> # |+> state with phase-flip noise
    >>> rho = pure_state_density_matrix([1, 1])
    >>> rho_noisy = phase_flip_channel(rho, 0.1)
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Probability p must be in [0, 1], got {p}")

    kraus_ops = [
        np.sqrt(1 - p) * _PAULI_I,  # No error
        np.sqrt(p) * _PAULI_Z        # Phase flip
    ]

    return apply_channel(rho, kraus_ops)


def bit_phase_flip_channel(rho: Union[DensityMatrix, np.ndarray], p: float) -> DensityMatrix:
    """
    Bit-phase-flip channel: applies sigma_y with probability p.

    Physical meaning: combined bit and phase flip (sigma_y = i * sigma_x * sigma_z)

    Kraus operators:
    - K0 = sqrt(1-p) * I
    - K1 = sqrt(p) * sigma_y

    Parameters
    ----------
    rho : DensityMatrix or np.ndarray
        Input state
    p : float
        Error probability (0 <= p <= 1)

    Returns
    -------
    DensityMatrix
        State after bit-phase-flip noise
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Probability p must be in [0, 1], got {p}")

    kraus_ops = [
        np.sqrt(1 - p) * _PAULI_I,  # No error
        np.sqrt(p) * _PAULI_Y        # Y error
    ]

    return apply_channel(rho, kraus_ops)


def depolarizing_channel(rho: Union[DensityMatrix, np.ndarray], p: float) -> DensityMatrix:
    """
    Depolarizing channel: applies random Pauli error.

    Physical meaning: with probability p, qubit is replaced by maximally mixed state

    Result: rho -> (1-p) * rho + p * I/2

    Kraus operators:
    - K0 = sqrt(1 - 3p/4) * I
    - K1 = sqrt(p/4) * sigma_x
    - K2 = sqrt(p/4) * sigma_y
    - K3 = sqrt(p/4) * sigma_z

    Parameters
    ----------
    rho : DensityMatrix or np.ndarray
        Input state
    p : float
        Depolarizing probability (0 <= p <= 4/3)

    Returns
    -------
    DensityMatrix
        State after depolarizing noise

    Examples
    --------
    >>> # Pure state becoming mixed
    >>> rho = pure_state_density_matrix([1, 0])
    >>> rho_noisy = depolarizing_channel(rho, 0.1)
    >>> rho_noisy.purity()  # Less than 1.0
    """
    if not 0 <= p <= 4/3:
        raise ValueError(f"Depolarizing parameter p must be in [0, 4/3], got {p}")

    kraus_ops = [
        np.sqrt(1 - 3*p/4) * _PAULI_I,  # No error
        np.sqrt(p/4) * _PAULI_X,         # X error
        np.sqrt(p/4) * _PAULI_Y,         # Y error
        np.sqrt(p/4) * _PAULI_Z          # Z error
    ]

    return apply_channel(rho, kraus_ops)


def amplitude_damping_channel(rho: Union[DensityMatrix, np.ndarray], gamma: float) -> DensityMatrix:
    """
    Amplitude damping channel: models energy relaxation (T1 decay).

    Physical meaning: |1> decays to |0> (energy loss to environment)
    - |0> remains |0> (ground state is stable)
    - |1> -> sqrt(1-gamma) |1> + sqrt(gamma) |0>

    Kraus operators:
    - K0 = [[1, 0], [0, sqrt(1-gamma)]]
    - K1 = [[0, sqrt(gamma)], [0, 0]]

    Parameters
    ----------
    rho : DensityMatrix or np.ndarray
        Input state
    gamma : float
        Damping parameter (0 <= gamma <= 1)
        gamma = 1 - exp(-t/T1) for time t

    Returns
    -------
    DensityMatrix
        State after amplitude damping

    Examples
    --------
    >>> # Excited state |1> decaying
    >>> rho = pure_state_density_matrix([0, 1])
    >>> rho_decayed = amplitude_damping_channel(rho, 0.5)
    >>> # Population transferred from |1> to |0>
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"Gamma must be in [0, 1], got {gamma}")

    K0 = np.array([[1, 0],
                   [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)],
                   [0, 0]], dtype=complex)

    kraus_ops = [K0, K1]

    return apply_channel(rho, kraus_ops)


def phase_damping_channel(rho: Union[DensityMatrix, np.ndarray], lambda_: float) -> DensityMatrix:
    """
    Phase damping channel: models pure dephasing (T2 decay without energy loss).

    Physical meaning: loss of coherence (off-diagonal elements) without population change
    - Diagonal elements (populations) unchanged
    - Off-diagonal elements (coherences) decay

    Kraus operators:
    - K0 = [[1, 0], [0, sqrt(1-lambda)]]
    - K1 = [[0, 0], [0, sqrt(lambda)]]

    Parameters
    ----------
    rho : DensityMatrix or np.ndarray
        Input state
    lambda_ : float
        Dephasing parameter (0 <= lambda <= 1)
        lambda = 1 - exp(-t/T2) for time t

    Returns
    -------
    DensityMatrix
        State after phase damping

    Examples
    --------
    >>> # Superposition |+> losing coherence
    >>> rho = pure_state_density_matrix([1, 1])
    >>> rho_dephased = phase_damping_channel(rho, 0.5)
    >>> # Off-diagonal elements reduced
    """
    if not 0 <= lambda_ <= 1:
        raise ValueError(f"Lambda must be in [0, 1], got {lambda_}")

    K0 = np.array([[1, 0],
                   [0, np.sqrt(1 - lambda_)]], dtype=complex)
    K1 = np.array([[0, 0],
                   [0, np.sqrt(lambda_)]], dtype=complex)

    kraus_ops = [K0, K1]

    return apply_channel(rho, kraus_ops)


def combined_ad_pd_channel(rho: Union[DensityMatrix, np.ndarray],
                            gamma: float, lambda_: float) -> DensityMatrix:
    """
    Combined amplitude and phase damping.

    Models realistic decoherence with both T1 and T2 processes.

    Parameters
    ----------
    rho : DensityMatrix or np.ndarray
        Input state
    gamma : float
        Amplitude damping (T1)
    lambda_ : float
        Phase damping (T2)

    Returns
    -------
    DensityMatrix
        State after combined decoherence

    Note
    ----
    Physical constraint: T2 <= 2*T1, which means lambda >= gamma/2
    """
    # Apply amplitude damping first
    rho_after_ad = amplitude_damping_channel(rho, gamma)
    # Then phase damping
    rho_final = phase_damping_channel(rho_after_ad, lambda_)
    return rho_final


def pauli_channel(rho: Union[DensityMatrix, np.ndarray],
                  px: float, py: float, pz: float) -> DensityMatrix:
    """
    General Pauli channel with specified error rates.

    Applies sigma_x with probability px, sigma_y with py, sigma_z with pz.

    Parameters
    ----------
    rho : DensityMatrix or np.ndarray
        Input state
    px, py, pz : float
        Probabilities for X, Y, Z errors
        Must satisfy: px + py + pz <= 1

    Returns
    -------
    DensityMatrix
        State after Pauli errors
    """
    if not 0 <= px + py + pz <= 1:
        raise ValueError(f"Sum of probabilities must be <= 1, got {px + py + pz}")

    p0 = 1 - px - py - pz  # No error

    kraus_ops = [
        np.sqrt(p0) * _PAULI_I,
        np.sqrt(px) * _PAULI_X,
        np.sqrt(py) * _PAULI_Y,
        np.sqrt(pz) * _PAULI_Z
    ]

    return apply_channel(rho, kraus_ops)


# Self-test when run directly
if __name__ == "__main__":
    from .density_matrix import pure_state_density_matrix

    print("=" * 70)
    print("QUANTUM CHANNELS MODULE - SELF TEST")
    print("=" * 70)

    # Test 1: Bit-flip channel
    print("\n1. Bit-Flip Channel")
    rho_0 = pure_state_density_matrix([1, 0])  # |0>
    print(f"   Initial: {rho_0}")
    rho_bf = bit_flip_channel(rho_0, 0.1)
    print(f"   After 10% bit-flip: {rho_bf}")
    print(f"   <0|rho|0> = {rho_bf.matrix[0,0].real:.4f} (expected: 0.90)")
    print(f"   <1|rho|1> = {rho_bf.matrix[1,1].real:.4f} (expected: 0.10)")

    # Test 2: Phase-flip channel
    print("\n2. Phase-Flip Channel")
    rho_plus = pure_state_density_matrix([1, 1])  # |+>
    print(f"   Initial: {rho_plus}")
    print(f"   Initial Bloch: {rho_plus.bloch_vector()}")
    rho_pf = phase_flip_channel(rho_plus, 0.2)
    print(f"   After 20% phase-flip: {rho_pf}")
    x, y, z = rho_pf.bloch_vector()
    print(f"   Bloch vector: ({x:.4f}, {y:.4f}, {z:.4f})")
    print(f"   x-coordinate reduced: {x:.4f} (expected: ~0.6)")

    # Test 3: Depolarizing channel
    print("\n3. Depolarizing Channel")
    rho_0 = pure_state_density_matrix([1, 0])
    print(f"   Initial purity: {rho_0.purity():.6f}")
    rho_depol = depolarizing_channel(rho_0, 0.3)
    print(f"   After 30% depolarizing: purity = {rho_depol.purity():.6f}")
    print(f"   Expected: less than 1.0 (mixed state)")

    # Test 4: Amplitude damping
    print("\n4. Amplitude Damping Channel (T1)")
    rho_1 = pure_state_density_matrix([0, 1])  # |1>
    print(f"   Initial: |1> state")
    print(f"   Population in |1>: {rho_1.matrix[1,1].real:.4f}")
    rho_ad = amplitude_damping_channel(rho_1, 0.5)
    print(f"   After gamma=0.5 damping:")
    print(f"   Population in |0>: {rho_ad.matrix[0,0].real:.4f}")
    print(f"   Population in |1>: {rho_ad.matrix[1,1].real:.4f}")
    print(f"   Energy lost to environment!")

    # Test 5: Phase damping
    print("\n5. Phase Damping Channel (T2)")
    rho_plus = pure_state_density_matrix([1, 1])
    print(f"   Initial: |+> state")
    print(f"   Coherence: {abs(rho_plus.matrix[0,1]):.4f}")
    rho_pd = phase_damping_channel(rho_plus, 0.5)
    print(f"   After lambda=0.5 dephasing:")
    print(f"   Coherence: {abs(rho_pd.matrix[0,1]):.4f}")
    print(f"   Populations unchanged:")
    print(f"   <0|rho|0> = {rho_pd.matrix[0,0].real:.4f}")
    print(f"   <1|rho|1> = {rho_pd.matrix[1,1].real:.4f}")

    # Test 6: Kraus completeness
    print("\n6. Kraus Completeness Verification")
    # Bit-flip
    K_bf = [np.sqrt(0.9) * _PAULI_I, np.sqrt(0.1) * _PAULI_X]
    print(f"   Bit-flip Kraus complete: {verify_kraus_completeness(K_bf)}")

    # Amplitude damping
    gamma = 0.3
    K_ad = [
        np.array([[1, 0], [0, np.sqrt(1-gamma)]]),
        np.array([[0, np.sqrt(gamma)], [0, 0]])
    ]
    print(f"   Amplitude damping Kraus complete: {verify_kraus_completeness(K_ad)}")

    # Test 7: Channel composition
    print("\n7. Combined T1 + T2 Decoherence")
    rho_plus = pure_state_density_matrix([1, 1])
    print(f"   Initial purity: {rho_plus.purity():.6f}")
    rho_combined = combined_ad_pd_channel(rho_plus, gamma=0.2, lambda_=0.3)
    print(f"   After T1 + T2 decay: purity = {rho_combined.purity():.6f}")
    x, y, z = rho_combined.bloch_vector()
    print(f"   Bloch vector: ({x:.4f}, {y:.4f}, {z:.4f})")

    print("\n" + "=" * 70)
    print("ALL CHANNEL TESTS COMPLETED!")
    print("=" * 70)
