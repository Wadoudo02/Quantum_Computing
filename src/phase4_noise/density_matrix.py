# -*- coding: utf-8 -*-
"""
Density Matrix Operations for Mixed Quantum States

Implements density matrix formalism for representing both pure and mixed quantum states.
This is essential for describing quantum systems that interact with their environment.

Based on Imperial College Notes - Section 4.1: Density matrices

Key Concepts:
- Density matrix: rho = sum_i p_i |psi_i><psi_i|
- Properties: Hermitian, unit trace, positive semi-definite
- Purity: Tr(rho^2) in [1/d, 1] where d is dimension
- Fidelity: F(rho, sigma) measures closeness of quantum states

Author: Wadoud Charbak
"""

import numpy as np
from typing import List, Tuple, Union, Optional
import warnings


class DensityMatrix:
    """
    Represents a quantum state using the density matrix formalism.

    Can represent both pure states (rho = |psi><psi|) and mixed states (statistical mixtures).

    Attributes
    ----------
    matrix : np.ndarray
        The density matrix (complex, Hermitian, positive semi-definite)
    n_qubits : int
        Number of qubits (dimension = 2^n_qubits)

    Examples
    --------
    >>> # Pure state |0>
    >>> rho = DensityMatrix.from_state_vector(np.array([1, 0]))
    >>> print(rho.is_pure())
    True

    >>> # Mixed state: 50% |0>, 50% |1>
    >>> rho_mixed = DensityMatrix.from_mixed_state(
    ...     [np.array([1, 0]), np.array([0, 1])],
    ...     [0.5, 0.5]
    ... )
    >>> print(rho_mixed.is_pure())
    False
    """

    def __init__(self, matrix: np.ndarray):
        """
        Initialize density matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Complex matrix representing the density operator

        Raises
        ------
        ValueError
            If matrix doesn't satisfy density matrix properties
        """
        self.matrix = np.array(matrix, dtype=complex)

        # Validate density matrix properties
        self._validate()

        # Compute number of qubits
        dim = self.matrix.shape[0]
        if not (dim & (dim - 1) == 0):  # Check if power of 2
            warnings.warn(f"Dimension {dim} is not a power of 2. Not a standard qubit system.")
            self.n_qubits = None
        else:
            self.n_qubits = int(np.log2(dim))

    def _validate(self, tolerance=1e-10):
        """Validate density matrix properties."""
        # Square matrix
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Density matrix must be square")

        # Hermitian
        if not np.allclose(self.matrix, self.matrix.conj().T, atol=tolerance):
            raise ValueError("Density matrix must be Hermitian")

        # Unit trace
        trace = np.trace(self.matrix)
        if not np.isclose(trace, 1.0, atol=tolerance):
            raise ValueError(f"Density matrix must have unit trace, got {trace}")

        # Positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        if np.any(eigenvalues < -tolerance):
            raise ValueError(f"Density matrix must be positive semi-definite, got min eigenvalue {eigenvalues.min()}")

    @classmethod
    def from_state_vector(cls, state: np.ndarray) -> 'DensityMatrix':
        """
        Create density matrix from pure state vector.

        rho = |psi><psi|

        Parameters
        ----------
        state : np.ndarray
            Normalized state vector

        Returns
        -------
        DensityMatrix
            Pure state density matrix

        Examples
        --------
        >>> state = np.array([1, 0])  # |0>
        >>> rho = DensityMatrix.from_state_vector(state)
        """
        state = np.array(state, dtype=complex)
        state = state / np.linalg.norm(state)  # Normalize
        matrix = np.outer(state, state.conj())
        return cls(matrix)

    @classmethod
    def from_mixed_state(cls, states: List[np.ndarray], probabilities: List[float]) -> 'DensityMatrix':
        """
        Create density matrix from statistical mixture.

        rho = sum_i p_i |psi_i><psi_i|

        Parameters
        ----------
        states : List[np.ndarray]
            List of state vectors
        probabilities : List[float]
            Probabilities for each state (must sum to 1)

        Returns
        -------
        DensityMatrix
            Mixed state density matrix

        Examples
        --------
        >>> # Maximally mixed state: 50% |0> + 50% |1>
        >>> rho = DensityMatrix.from_mixed_state(
        ...     [np.array([1, 0]), np.array([0, 1])],
        ...     [0.5, 0.5]
        ... )
        """
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError(f"Probabilities must sum to 1, got {sum(probabilities)}")

        # Normalize all states
        states = [s / np.linalg.norm(s) for s in states]

        # Build density matrix
        matrix = sum(p * np.outer(s, s.conj()) for s, p in zip(states, probabilities))
        return cls(matrix)

    @classmethod
    def maximally_mixed(cls, dimension: int) -> 'DensityMatrix':
        """
        Create maximally mixed state.

        rho = I/d (completely random)

        Parameters
        ----------
        dimension : int
            Hilbert space dimension

        Returns
        -------
        DensityMatrix
            Maximally mixed state
        """
        matrix = np.eye(dimension, dtype=complex) / dimension
        return cls(matrix)

    def purity(self) -> float:
        """
        Calculate purity: Tr(rho^2).

        Returns
        -------
        float
            Purity in [1/d, 1]
            - 1 for pure states
            - 1/d for maximally mixed states

        Examples
        --------
        >>> rho_pure = DensityMatrix.from_state_vector([1, 0])
        >>> rho_pure.purity()
        1.0
        >>> rho_mixed = DensityMatrix.maximally_mixed(2)
        >>> rho_mixed.purity()
        0.5
        """
        return float(np.real(np.trace(self.matrix @ self.matrix)))

    def is_pure(self, tolerance=1e-10) -> bool:
        """
        Check if state is pure.

        Parameters
        ----------
        tolerance : float
            Numerical tolerance

        Returns
        -------
        bool
            True if Tr(rho^2) ~ 1
        """
        return np.isclose(self.purity(), 1.0, atol=tolerance)

    def von_neumann_entropy(self) -> float:
        """
        Calculate von Neumann entropy: S(rho) = -Tr(rho log_2 rho).

        Returns
        -------
        float
            Entropy in bits
            - 0 for pure states
            - log_2(d) for maximally mixed states

        Examples
        --------
        >>> rho_pure = DensityMatrix.from_state_vector([1, 0])
        >>> rho_pure.von_neumann_entropy()
        0.0
        >>> rho_mixed = DensityMatrix.maximally_mixed(2)
        >>> rho_mixed.von_neumann_entropy()
        1.0
        """
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        # Filter out zero/negative eigenvalues (numerical errors)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def bloch_vector(self) -> Optional[Tuple[float, float, float]]:
        """
        Extract Bloch vector for single-qubit state.

        rho = (I + r.sigma)/2 where r = (x, y, z)

        Returns
        -------
        Tuple[float, float, float] or None
            (x, y, z) coordinates on Bloch sphere, or None if not single qubit

        Examples
        --------
        >>> rho = DensityMatrix.from_state_vector([1, 0])  # |0>
        >>> rho.bloch_vector()
        (0.0, 0.0, 1.0)
        >>> rho = DensityMatrix.from_state_vector([1, 1])  # |+>
        >>> rho.bloch_vector()
        (1.0, 0.0, 0.0)
        """
        if self.matrix.shape != (2, 2):
            return None

        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Extract coordinates: x = Tr(rho sigma_x), etc.
        x = float(np.real(np.trace(self.matrix @ sigma_x)))
        y = float(np.real(np.trace(self.matrix @ sigma_y)))
        z = float(np.real(np.trace(self.matrix @ sigma_z)))

        return (x, y, z)

    def expectation_value(self, operator: np.ndarray) -> complex:
        """
        Calculate expectation value: <A> = Tr(rho A).

        Parameters
        ----------
        operator : np.ndarray
            Observable operator

        Returns
        -------
        complex
            Expectation value
        """
        return np.trace(self.matrix @ operator)

    def apply_unitary(self, U: np.ndarray) -> 'DensityMatrix':
        """
        Apply unitary transformation: rho -> U rho U_dagger.

        Parameters
        ----------
        U : np.ndarray
            Unitary matrix

        Returns
        -------
        DensityMatrix
            Transformed state
        """
        new_matrix = U @ self.matrix @ U.conj().T
        return DensityMatrix(new_matrix)

    def partial_trace(self, keep_indices: List[int]) -> 'DensityMatrix':
        """
        Compute partial trace over specified subsystems.

        Parameters
        ----------
        keep_indices : List[int]
            Indices of qubits to keep (0-indexed)

        Returns
        -------
        DensityMatrix
            Reduced density matrix

        Examples
        --------
        >>> # Bell state |Phi+> = (|00> + |11>)/sqrt(2)
        >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
        >>> rho = DensityMatrix.from_state_vector(bell)
        >>> rho_A = rho.partial_trace([0])  # Trace out qubit 1
        >>> rho_A.purity()
        0.5  # Maximally mixed!
        """
        if self.n_qubits is None:
            raise ValueError("Partial trace only defined for qubit systems")

        # Simple implementation for 2-qubit case
        if self.n_qubits == 2:
            rho_reshaped = self.matrix.reshape(2, 2, 2, 2)
            if keep_indices == [0]:
                # Trace out qubit 1
                result = np.trace(rho_reshaped, axis1=1, axis2=3)
            elif keep_indices == [1]:
                # Trace out qubit 0
                result = np.trace(rho_reshaped, axis1=0, axis2=2)
            else:
                raise ValueError("Invalid keep_indices for 2-qubit system")
            return DensityMatrix(result)
        else:
            raise NotImplementedError("Partial trace only implemented for 2-qubit systems")

    def __repr__(self) -> str:
        """String representation."""
        pure_str = "Pure" if self.is_pure() else "Mixed"
        return f"DensityMatrix({pure_str}, dim={self.matrix.shape[0]}, purity={self.purity():.4f})"

    def __str__(self) -> str:
        """Detailed string representation."""
        lines = [f"Density Matrix ({self.matrix.shape[0]}x{self.matrix.shape[0]}):"]
        lines.append(f"  Pure: {self.is_pure()}")
        lines.append(f"  Purity: {self.purity():.6f}")
        lines.append(f"  Entropy: {self.von_neumann_entropy():.6f} bits")
        if self.n_qubits == 1:
            x, y, z = self.bloch_vector()
            lines.append(f"  Bloch: ({x:.4f}, {y:.4f}, {z:.4f})")
        return "\n".join(lines)


# Standalone functions for convenience
def pure_state_density_matrix(state: np.ndarray) -> DensityMatrix:
    """Create density matrix from pure state vector."""
    return DensityMatrix.from_state_vector(state)


def mixed_state_density_matrix(states: List[np.ndarray], probabilities: List[float]) -> DensityMatrix:
    """Create density matrix from mixed state."""
    return DensityMatrix.from_mixed_state(states, probabilities)


def purity(rho: Union[DensityMatrix, np.ndarray]) -> float:
    """Calculate purity Tr(rho^2)."""
    if isinstance(rho, DensityMatrix):
        return rho.purity()
    return float(np.real(np.trace(rho @ rho)))


def is_pure(rho: Union[DensityMatrix, np.ndarray], tolerance=1e-10) -> bool:
    """Check if state is pure."""
    if isinstance(rho, DensityMatrix):
        return rho.is_pure(tolerance)
    return np.isclose(purity(rho), 1.0, atol=tolerance)


def fidelity(rho1: Union[DensityMatrix, np.ndarray], rho2: Union[DensityMatrix, np.ndarray]) -> float:
    """
    Calculate fidelity between two density matrices.

    F(rho, sigma) = Tr(sqrt(sqrt(rho) sigma sqrt(rho)))^2

    For pure states: F = |<psi|phi>|^2

    Parameters
    ----------
    rho1, rho2 : DensityMatrix or np.ndarray
        Density matrices to compare

    Returns
    -------
    float
        Fidelity in [0, 1]

    Examples
    --------
    >>> rho1 = pure_state_density_matrix([1, 0])  # |0>
    >>> rho2 = pure_state_density_matrix([0, 1])  # |1>
    >>> fidelity(rho1, rho2)
    0.0
    >>> fidelity(rho1, rho1)
    1.0
    """
    # Extract matrices
    m1 = rho1.matrix if isinstance(rho1, DensityMatrix) else rho1
    m2 = rho2.matrix if isinstance(rho2, DensityMatrix) else rho2

    # Compute sqrt(rho1)
    eigvals1, eigvecs1 = np.linalg.eigh(m1)
    eigvals1 = np.maximum(eigvals1, 0)  # Avoid numerical errors
    sqrt_rho1 = eigvecs1 @ np.diag(np.sqrt(eigvals1)) @ eigvecs1.conj().T

    # Compute sqrt(rho1) sigma sqrt(rho1)
    M = sqrt_rho1 @ m2 @ sqrt_rho1

    # Compute sqrt(M)
    eigvals_M, eigvecs_M = np.linalg.eigh(M)
    eigvals_M = np.maximum(eigvals_M, 0)
    sqrt_M = eigvecs_M @ np.diag(np.sqrt(eigvals_M)) @ eigvecs_M.conj().T

    # F = (Tr(sqrt(M)))^2
    trace_sqrt_M = np.trace(sqrt_M)
    F = np.real(trace_sqrt_M) ** 2

    return float(np.clip(F, 0, 1))  # Ensure [0, 1]


def trace_distance(rho1: Union[DensityMatrix, np.ndarray], rho2: Union[DensityMatrix, np.ndarray]) -> float:
    """
    Calculate trace distance between density matrices.

    D(rho, sigma) = (1/2) Tr(|rho - sigma|)

    where |A| = sqrt(A_dagger A) is the matrix absolute value.

    Parameters
    ----------
    rho1, rho2 : DensityMatrix or np.ndarray
        Density matrices

    Returns
    -------
    float
        Trace distance in [0, 1]
    """
    m1 = rho1.matrix if isinstance(rho1, DensityMatrix) else rho1
    m2 = rho2.matrix if isinstance(rho2, DensityMatrix) else rho2

    diff = m1 - m2
    eigenvalues = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def bloch_vector(rho: Union[DensityMatrix, np.ndarray]) -> Optional[Tuple[float, float, float]]:
    """Extract Bloch vector from single-qubit density matrix."""
    if isinstance(rho, DensityMatrix):
        return rho.bloch_vector()

    if rho.shape != (2, 2):
        return None

    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    x = float(np.real(np.trace(rho @ sigma_x)))
    y = float(np.real(np.trace(rho @ sigma_y)))
    z = float(np.real(np.trace(rho @ sigma_z)))

    return (x, y, z)


# Self-test when run directly
if __name__ == "__main__":
    print("=" * 70)
    print("DENSITY MATRIX MODULE - SELF TEST")
    print("=" * 70)

    # Test 1: Pure state
    print("\n1. Pure State |0>")
    state_0 = np.array([1, 0])
    rho_0 = pure_state_density_matrix(state_0)
    print(rho_0)
    print(f"   Matrix:\n{rho_0.matrix}")

    # Test 2: Pure state |+>
    print("\n2. Pure State |+>")
    state_plus = np.array([1, 1]) / np.sqrt(2)
    rho_plus = pure_state_density_matrix(state_plus)
    print(rho_plus)
    x, y, z = rho_plus.bloch_vector()
    print(f"   Bloch vector: ({x:.4f}, {y:.4f}, {z:.4f})")
    print(f"   Expected: (1.0000, 0.0000, 0.0000)")

    # Test 3: Maximally mixed state
    print("\n3. Maximally Mixed State")
    rho_mixed = DensityMatrix.maximally_mixed(2)
    print(rho_mixed)
    print(f"   Matrix:\n{rho_mixed.matrix}")

    # Test 4: Mixed state (50% |0>, 50% |1>)
    print("\n4. Mixed State: 50% |0> + 50% |1>")
    rho_mix = mixed_state_density_matrix(
        [np.array([1, 0]), np.array([0, 1])],
        [0.5, 0.5]
    )
    print(rho_mix)

    # Test 5: Fidelity
    print("\n5. Fidelity Tests")
    print(f"   F(|0>, |0>) = {fidelity(rho_0, rho_0):.6f} (expected: 1.0)")
    rho_1 = pure_state_density_matrix(np.array([0, 1]))
    print(f"   F(|0>, |1>) = {fidelity(rho_0, rho_1):.6f} (expected: 0.0)")
    print(f"   F(|0>, |+>) = {fidelity(rho_0, rho_plus):.6f} (expected: 0.5)")

    # Test 6: Trace distance
    print("\n6. Trace Distance Tests")
    print(f"   D(|0>, |0>) = {trace_distance(rho_0, rho_0):.6f} (expected: 0.0)")
    print(f"   D(|0>, |1>) = {trace_distance(rho_0, rho_1):.6f} (expected: 1.0)")

    # Test 7: Bell state partial trace
    print("\n7. Bell State Partial Trace")
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |Phi+>
    rho_bell = pure_state_density_matrix(bell_state)
    print(f"   Bell state purity: {rho_bell.purity():.6f} (expected: 1.0)")
    rho_A = rho_bell.partial_trace([0])
    print(f"   Reduced state purity: {rho_A.purity():.6f} (expected: 0.5)")
    print(f"   Reduced state entropy: {rho_A.von_neumann_entropy():.6f} bits (expected: 1.0)")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED!")
    print("=" * 70)
