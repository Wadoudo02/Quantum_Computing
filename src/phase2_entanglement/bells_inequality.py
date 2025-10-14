"""
Bell's Inequality and CHSH Test
================================

Implementation of the CHSH (Clauser-Horne-Shimony-Holt) inequality test,
demonstrating the fundamental non-locality of quantum mechanics.

The CHSH inequality states that for any local hidden variable theory:
    |E(a,b) + E(a,b') + E(a',b) - E(a',b')| ≤ 2

where E(a,b) is the correlation between measurements in directions a and b.

Quantum mechanics can violate this, achieving values up to 2√2 ≈ 2.828,
proving that quantum entanglement exhibits genuine non-local correlations.

Reference: Imperial College notes on Bell's inequality
"""

import numpy as np
from typing import Tuple, List, Dict

# Handle both package and direct imports
try:
    from .bell_states import BellState, bell_phi_plus
except ImportError:
    from bell_states import BellState, bell_phi_plus


# ============================================================================
# Measurement Basis Helpers
# ============================================================================

def rotation_basis(theta: float) -> np.ndarray:
    """
    Create a measurement basis rotated by angle theta on the Bloch sphere.

    This corresponds to measuring in a basis that's rotated from |0⟩/|1⟩
    by angle theta around the Y-axis.

    Parameters
    ----------
    theta : float
        Rotation angle in radians

    Returns
    -------
    np.ndarray
        2x2 unitary rotation matrix
    """
    cos_half = np.cos(theta / 2)
    sin_half = np.sin(theta / 2)

    return np.array([
        [cos_half, -sin_half],
        [sin_half, cos_half]
    ], dtype=complex)


# ============================================================================
# Correlation Measurements
# ============================================================================

def measure_correlation(
    state: BellState,
    angle_a: float,
    angle_b: float,
    shots: int = 10000
) -> float:
    """
    Measure quantum correlation E(a,b) between two qubits.

    The correlation is defined as:
        E(a,b) = P(same) - P(different)

    where measurements are performed in rotated bases specified by angles.

    Parameters
    ----------
    state : BellState
        The two-qubit state to measure
    angle_a : float
        Measurement angle for qubit A (radians)
    angle_b : float
        Measurement angle for qubit B (radians)
    shots : int
        Number of measurement samples

    Returns
    -------
    float
        Correlation value between -1 and +1
    """
    # Create measurement bases
    basis_a = rotation_basis(angle_a)
    basis_b = rotation_basis(angle_b)

    # Perform measurements
    results = state.measure_in_basis(basis_a, basis_b, shots=shots)

    # Calculate correlation: E = P(same) - P(different)
    same = np.sum(results[:, 0] == results[:, 1])
    different = shots - same

    correlation = (same - different) / shots

    return correlation


def measure_correlation_exact(
    state: BellState,
    angle_a: float,
    angle_b: float
) -> float:
    """
    Calculate exact quantum correlation analytically (no sampling).

    For a Bell state |Φ+⟩, the correlation is:
        E(a,b) = cos(angle_a - angle_b)

    This gives the theoretical prediction without measurement noise.

    Parameters
    ----------
    state : BellState
        The two-qubit state
    angle_a : float
        Measurement angle for qubit A (radians)
    angle_b : float
        Measurement angle for qubit B (radians)

    Returns
    -------
    float
        Exact correlation value
    """
    # For |Φ+⟩ state, correlation is cos(theta_a - theta_b)
    # This is derived from the section on Bell's inequality
    return np.cos(angle_a - angle_b)


# ============================================================================
# CHSH Inequality
# ============================================================================

def compute_chsh_value(
    state: BellState,
    a: float,
    a_prime: float,
    b: float,
    b_prime: float,
    shots: int = 10000,
    exact: bool = False
) -> float:
    """
    Compute the CHSH parameter S for given measurement angles.

    S = E(a,b) + E(a,b') + E(a',b) - E(a',b')

    Classical bound: |S| ≤ 2
    Quantum maximum: |S| ≤ 2√2 ≈ 2.828 (Tsirelson's bound)

    Parameters
    ----------
    state : BellState
        The entangled state to test
    a, a_prime : float
        Alice's two measurement angles (radians)
    b, b_prime : float
        Bob's two measurement angles (radians)
    shots : int
        Number of measurements per correlation
    exact : bool
        If True, use exact calculation; if False, simulate measurements

    Returns
    -------
    float
        The CHSH parameter S
    """
    if exact:
        # Exact quantum prediction
        E_ab = measure_correlation_exact(state, a, b)
        E_ab_prime = measure_correlation_exact(state, a, b_prime)
        E_a_prime_b = measure_correlation_exact(state, a_prime, b)
        E_a_prime_b_prime = measure_correlation_exact(state, a_prime, b_prime)
    else:
        # Simulated measurements
        E_ab = measure_correlation(state, a, b, shots)
        E_ab_prime = measure_correlation(state, a, b_prime, shots)
        E_a_prime_b = measure_correlation(state, a_prime, b, shots)
        E_a_prime_b_prime = measure_correlation(state, a_prime, b_prime, shots)

    # CHSH parameter
    S = E_ab + E_ab_prime + E_a_prime_b - E_a_prime_b_prime

    return S


def optimal_chsh_angles() -> Tuple[float, float, float, float]:
    """
    Return the optimal angles for maximum CHSH violation.

    These angles achieve the maximum quantum value of 2√2.

    Returns
    -------
    a, a_prime, b, b_prime : float
        Optimal measurement angles in radians
    """
    # Standard optimal angles for CHSH test
    a = 0
    a_prime = np.pi / 2
    b = np.pi / 4
    b_prime = -np.pi / 4

    return a, a_prime, b, b_prime


def classical_bound() -> float:
    """
    Return the classical (local hidden variable) bound for CHSH.

    Returns
    -------
    float
        Classical maximum: 2
    """
    return 2.0


def quantum_bound() -> float:
    """
    Return the quantum mechanical bound for CHSH (Tsirelson's bound).

    Returns
    -------
    float
        Quantum maximum: 2√2 ≈ 2.828
    """
    return 2 * np.sqrt(2)


# ============================================================================
# Demonstration and Analysis
# ============================================================================

def demonstrate_bell_violation(
    shots: int = 10000,
    num_trials: int = 100
) -> Dict[str, any]:
    """
    Demonstrate violation of Bell's inequality with statistics.

    This function runs multiple CHSH tests and collects statistics to show
    that quantum mechanics consistently violates the classical bound.

    Parameters
    ----------
    shots : int
        Number of measurements per correlation
    num_trials : int
        Number of independent CHSH tests to run

    Returns
    -------
    dict
        Results containing:
        - 'chsh_values': List of CHSH parameters from all trials
        - 'mean_chsh': Average CHSH value
        - 'std_chsh': Standard deviation
        - 'classical_bound': Classical limit (2.0)
        - 'quantum_bound': Quantum limit (2√2)
        - 'violation_rate': Fraction of trials violating classical bound
        - 'exact_value': Theoretical quantum prediction
    """
    # Create Bell state |Φ+⟩
    state = bell_phi_plus()

    # Get optimal angles
    a, a_prime, b, b_prime = optimal_chsh_angles()

    # Run multiple trials
    chsh_values = []
    for _ in range(num_trials):
        S = compute_chsh_value(state, a, a_prime, b, b_prime, shots=shots, exact=False)
        chsh_values.append(S)

    chsh_values = np.array(chsh_values)

    # Calculate statistics
    mean_chsh = np.mean(chsh_values)
    std_chsh = np.std(chsh_values)

    # Count violations
    violations = np.sum(np.abs(chsh_values) > 2.0)
    violation_rate = violations / num_trials

    # Exact quantum prediction
    exact_value = compute_chsh_value(state, a, a_prime, b, b_prime, exact=True)

    return {
        'chsh_values': chsh_values,
        'mean_chsh': mean_chsh,
        'std_chsh': std_chsh,
        'classical_bound': classical_bound(),
        'quantum_bound': quantum_bound(),
        'violation_rate': violation_rate,
        'exact_value': exact_value,
        'angles': (a, a_prime, b, b_prime)
    }


def scan_chsh_angles(
    angle_range: Tuple[float, float] = (0, 2 * np.pi),
    num_points: int = 50,
    shots: int = 5000
) -> Dict[str, np.ndarray]:
    """
    Scan through different measurement angles and compute CHSH values.

    Useful for visualizing how CHSH parameter varies with measurement choices.

    Parameters
    ----------
    angle_range : tuple
        (min_angle, max_angle) to scan
    num_points : int
        Number of angle values to test
    shots : int
        Measurements per correlation

    Returns
    -------
    dict
        Contains 'angles' and 'chsh_values' arrays
    """
    state = bell_phi_plus()

    angles = np.linspace(angle_range[0], angle_range[1], num_points)
    chsh_values = []

    # Fix a and a', vary b
    a = 0
    a_prime = np.pi / 2
    b_prime = -np.pi / 4

    for b in angles:
        S = compute_chsh_value(state, a, a_prime, b, b_prime, shots=shots, exact=True)
        chsh_values.append(S)

    return {
        'angles': angles,
        'chsh_values': np.array(chsh_values)
    }


# ============================================================================
# Educational Helpers
# ============================================================================

def explain_bell_violation():
    """
    Return a detailed explanation of Bell's inequality violation.

    Returns
    -------
    str
        Educational text explaining the significance
    """
    explanation = """
    Bell's Inequality and Quantum Non-Locality
    ===========================================

    What is Bell's Inequality?
    ---------------------------
    In 1964, John Bell proved that any theory based on local realism
    (where particles have definite properties and influence cannot exceed
    light speed) must satisfy certain mathematical inequalities.

    The CHSH Version:
    ----------------
    The CHSH inequality states:
        |E(a,b) + E(a,b') + E(a',b) - E(a',b')| ≤ 2

    Where:
    - E(a,b) is the correlation between measurements at angles a and b
    - a, a', b, b' are different measurement directions

    Classical Prediction:
    -------------------
    Any local hidden variable theory predicts |S| ≤ 2

    Quantum Prediction:
    ------------------
    Quantum mechanics predicts |S| can reach 2√2 ≈ 2.828
    This is called Tsirelson's bound.

    What This Means:
    ---------------
    When we measure |S| > 2, we prove that:
    1. Quantum mechanics is fundamentally non-local
    2. Entangled particles exhibit "spooky action at a distance"
    3. No local hidden variable theory can explain quantum mechanics

    This is one of the most profound discoveries in physics!
    """

    return explanation


def compare_classical_quantum() -> Dict[str, float]:
    """
    Direct comparison of classical vs quantum predictions.

    Returns
    -------
    dict
        Comparison data with classical and quantum values
    """
    a, a_prime, b, b_prime = optimal_chsh_angles()
    state = bell_phi_plus()

    quantum_value = compute_chsh_value(state, a, a_prime, b, b_prime, exact=True)

    return {
        'classical_maximum': 2.0,
        'quantum_prediction': quantum_value,
        'quantum_maximum': 2 * np.sqrt(2),
        'violation_amount': quantum_value - 2.0,
        'violation_percentage': ((quantum_value - 2.0) / 2.0) * 100
    }
