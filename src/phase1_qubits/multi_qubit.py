"""
Multi-Qubit System Classes
===========================

Classes for representing and manipulating systems of multiple qubits.

Architecture:
- Qubit: Single qubit (in qubit.py)
- TwoQubitSystem: Two qubits (this file)
- MultiQubitSystem: N qubits (general case)

Reference: Imperial Notes Section 1.4 (Bipartite Systems)
"""

import numpy as np
from typing import Tuple, Optional
from .qubit import Qubit


class TwoQubitSystem:
    """
    A system of two qubits represented as a 4-dimensional state vector.

    The state is represented in the computational basis:
    |ψ⟩ = c₀₀|00⟩ + c₀₁|01⟩ + c₁₀|10⟩ + c₁₁|11⟩

    where |c₀₀|² + |c₀₁|² + |c₁₀|² + |c₁₁|² = 1

    Attributes
    ----------
    state : np.ndarray
        Complex vector of shape (4,) in basis [|00⟩, |01⟩, |10⟩, |11⟩]

    Examples
    --------
    >>> # Create |00⟩ state
    >>> sys = TwoQubitSystem([1, 0, 0, 0])
    >>> print(sys)
    |00⟩

    >>> # Create Bell state (|00⟩ + |11⟩)/√2
    >>> sys = TwoQubitSystem([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    >>> print(sys.is_entangled())
    True
    """

    def __init__(self, state: list | np.ndarray, normalize: bool = True):
        """
        Initialize a two-qubit system.

        Parameters
        ----------
        state : list or np.ndarray
            4-element array [c₀₀, c₀₁, c₁₀, c₁₁]
        normalize : bool, optional
            Whether to normalize the state (default: True)

        Raises
        ------
        ValueError
            If state is not 4-dimensional
        """
        self.state = np.array(state, dtype=complex)

        if len(self.state) != 4:
            raise ValueError("Two-qubit state must be 4-dimensional")

        if normalize:
            self._normalize()

    def _normalize(self) -> None:
        """Normalize the state vector to unit length."""
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state = self.state / norm

    @classmethod
    def from_single_qubits(cls, qubit1: Qubit, qubit2: Qubit) -> 'TwoQubitSystem':
        """
        Create a two-qubit system from two separate qubits.

        Performs tensor product: |ψ⟩ ⊗ |φ⟩

        Parameters
        ----------
        qubit1 : Qubit
            First qubit
        qubit2 : Qubit
            Second qubit

        Returns
        -------
        TwoQubitSystem
            Combined system

        Examples
        --------
        >>> from .qubit import ket_0, ket_1
        >>> sys = TwoQubitSystem.from_single_qubits(ket_0(), ket_1())
        >>> print(sys)  # |01⟩
        """
        state = np.kron(qubit1.state, qubit2.state)
        return cls(state, normalize=False)

    @property
    def c00(self) -> complex:
        """Amplitude of |00⟩ component."""
        return self.state[0]

    @property
    def c01(self) -> complex:
        """Amplitude of |01⟩ component."""
        return self.state[1]

    @property
    def c10(self) -> complex:
        """Amplitude of |10⟩ component."""
        return self.state[2]

    @property
    def c11(self) -> complex:
        """Amplitude of |11⟩ component."""
        return self.state[3]

    def measure(self, shots: int = 1) -> np.ndarray:
        """
        Measure both qubits in computational basis.

        Parameters
        ----------
        shots : int, optional
            Number of measurements (default: 1)

        Returns
        -------
        np.ndarray
            Array of shape (shots, 2) with values 0 or 1
            Each row is [qubit1_outcome, qubit2_outcome]

        Examples
        --------
        >>> sys = TwoQubitSystem([1, 0, 0, 0])  # |00⟩
        >>> results = sys.measure(10)
        >>> print(results)  # All [0, 0]
        """
        # Calculate probabilities
        probabilities = np.abs(self.state) ** 2

        # Sample from the distribution
        outcomes_flat = np.random.choice(4, size=shots, p=probabilities)

        # Convert to binary representation
        outcomes = np.zeros((shots, 2), dtype=int)
        for i, outcome in enumerate(outcomes_flat):
            outcomes[i] = [outcome // 2, outcome % 2]

        return outcomes

    def measure_qubit(self, qubit_index: int, shots: int = 1) -> np.ndarray:
        """
        Measure only one qubit (partial measurement).

        Parameters
        ----------
        qubit_index : int
            Which qubit to measure (0 or 1)
        shots : int, optional
            Number of measurements

        Returns
        -------
        np.ndarray
            Array of measurement outcomes (0s and 1s)
        """
        if qubit_index == 0:
            # Measure first qubit
            prob_0 = np.abs(self.c00)**2 + np.abs(self.c01)**2
        elif qubit_index == 1:
            # Measure second qubit
            prob_0 = np.abs(self.c00)**2 + np.abs(self.c10)**2
        else:
            raise ValueError("qubit_index must be 0 or 1")

        prob_1 = 1 - prob_0
        return np.random.choice([0, 1], size=shots, p=[prob_0, prob_1])

    def reduced_density_matrix(self, qubit_index: int) -> np.ndarray:
        """
        Calculate reduced density matrix for one qubit.

        This traces out the other qubit, giving the state of
        a single qubit in the system.

        Parameters
        ----------
        qubit_index : int
            Which qubit to keep (0 or 1)

        Returns
        -------
        np.ndarray
            2×2 density matrix for the selected qubit

        Notes
        -----
        From Imperial Notes Section 4.1.1 (Reduced States)
        ρ_A = Tr_B(|ψ⟩⟨ψ|)

        Examples
        --------
        >>> # Bell state
        >>> sys = TwoQubitSystem([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        >>> rho = sys.reduced_density_matrix(0)
        >>> print(rho)  # [[0.5, 0], [0, 0.5]] - maximally mixed!
        """
        # Create density matrix |ψ⟩⟨ψ|
        rho_full = np.outer(self.state, np.conj(self.state))

        # Reshape to 2×2×2×2 (qubit1, qubit2, qubit1', qubit2')
        rho_reshaped = rho_full.reshape(2, 2, 2, 2)

        if qubit_index == 0:
            # Trace out qubit 2: sum over second index
            rho_reduced = np.trace(rho_reshaped, axis1=1, axis2=3)
        elif qubit_index == 1:
            # Trace out qubit 1: sum over first index
            rho_reduced = np.trace(rho_reshaped, axis1=0, axis2=2)
        else:
            raise ValueError("qubit_index must be 0 or 1")

        return rho_reduced

    def is_entangled(self, tolerance: float = 1e-10) -> bool:
        """
        Check if the two-qubit state is entangled.

        Uses Schmidt decomposition: a pure state is entangled if
        it cannot be written as |ψ⟩ ⊗ |φ⟩ (i.e., has Schmidt rank > 1).

        Returns
        -------
        bool
            True if state is entangled

        Notes
        -----
        From Imperial Notes Section 5.2 (Schmidt Decomposition)
        A state is separable iff it can be written as a product state.

        Examples
        --------
        >>> # Product state |01⟩ = |0⟩ ⊗ |1⟩
        >>> sys = TwoQubitSystem([0, 1, 0, 0])
        >>> print(sys.is_entangled())
        False

        >>> # Bell state (entangled)
        >>> sys = TwoQubitSystem([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        >>> print(sys.is_entangled())
        True
        """
        # Calculate reduced density matrix for first qubit
        rho = self.reduced_density_matrix(0)

        # If pure (rank 1), state is separable
        # If mixed (rank 2), state is entangled
        eigenvalues = np.linalg.eigvalsh(rho)

        # Count non-zero eigenvalues (Schmidt coefficients)
        schmidt_rank = np.sum(eigenvalues > tolerance)

        return schmidt_rank > 1

    def schmidt_decomposition(self, tolerance: float = 1e-10) -> Tuple[np.ndarray, list, list]:
        """
        Compute Schmidt decomposition of the state.

        Returns
        -------
        coefficients : np.ndarray
            Schmidt coefficients (singular values)
        basis_A : list of np.ndarray
            Orthonormal basis for qubit 1
        basis_B : list of np.ndarray
            Orthonormal basis for qubit 2

        Notes
        -----
        From Imperial Notes Section 5.2:
        |ψ⟩ = Σᵢ λᵢ |iₐ⟩ |iᵦ⟩
        where λᵢ are Schmidt coefficients (non-negative, sum to 1)

        Examples
        --------
        >>> sys = TwoQubitSystem([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        >>> coeffs, basis_A, basis_B = sys.schmidt_decomposition()
        >>> print(coeffs)  # [0.707, 0.707] for Bell state
        """
        # Reshape state to 2×2 matrix
        psi_matrix = self.state.reshape(2, 2)

        # Perform SVD: Ψ = U Λ V†
        U, singular_values, Vh = np.linalg.svd(psi_matrix)

        # Filter out near-zero coefficients
        mask = singular_values > tolerance
        coefficients = singular_values[mask]

        # Schmidt bases
        basis_A = [U[:, i] for i in range(len(coefficients))]
        basis_B = [Vh[i, :].conj() for i in range(len(coefficients))]

        return coefficients, basis_A, basis_B

    def entanglement_entropy(self) -> float:
        """
        Calculate von Neumann entropy of entanglement.

        Returns
        -------
        float
            Entanglement entropy E = -Σᵢ λᵢ² log₂(λᵢ²)
            E = 0 for separable states
            E = 1 for maximally entangled states (e.g., Bell states)

        Notes
        -----
        From Imperial Notes Section 5.1.1 (Entanglement Measures)
        """
        coefficients, _, _ = self.schmidt_decomposition()

        # Calculate entropy: -Σ λᵢ² log₂(λᵢ²)
        lambda_squared = coefficients ** 2

        # Filter out zeros to avoid log(0)
        nonzero = lambda_squared[lambda_squared > 1e-15]

        if len(nonzero) == 0:
            return 0.0

        entropy = -np.sum(nonzero * np.log2(nonzero))
        return entropy

    def copy(self) -> 'TwoQubitSystem':
        """Create a copy of this system."""
        return TwoQubitSystem(self.state.copy(), normalize=False)

    def __str__(self) -> str:
        """Human-readable representation in Dirac notation."""
        basis_labels = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

        # Check for pure basis states
        for i, (coeff, label) in enumerate(zip(self.state, basis_labels)):
            if np.abs(coeff - 1) < 1e-10 and np.all(np.abs(self.state) < 1e-10):
                return label

        # General case
        parts = []
        for coeff, label in zip(self.state, basis_labels):
            if np.abs(coeff) > 1e-10:
                if np.abs(np.imag(coeff)) < 1e-10:
                    parts.append(f"{np.real(coeff):.3f}{label}")
                else:
                    parts.append(f"({coeff:.3f}){label}")

        if not parts:
            return "0"

        result = parts[0]
        for part in parts[1:]:
            if part[0] == '-' or part[0] == '(':
                result += " " + part
            else:
                result += " + " + part

        return result

    def __repr__(self) -> str:
        """Programmer-friendly representation."""
        return f"TwoQubitSystem(state={self.state})"


# Convenience functions for common two-qubit states

def two_ket_00() -> TwoQubitSystem:
    """Create |00⟩ state."""
    return TwoQubitSystem([1, 0, 0, 0])


def two_ket_01() -> TwoQubitSystem:
    """Create |01⟩ state."""
    return TwoQubitSystem([0, 1, 0, 0])


def two_ket_10() -> TwoQubitSystem:
    """Create |10⟩ state."""
    return TwoQubitSystem([0, 0, 1, 0])


def two_ket_11() -> TwoQubitSystem:
    """Create |11⟩ state."""
    return TwoQubitSystem([0, 0, 0, 1])


def bell_phi_plus() -> TwoQubitSystem:
    """
    Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2.

    Maximally entangled state.
    """
    return TwoQubitSystem([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])


def bell_phi_minus() -> TwoQubitSystem:
    """
    Create Bell state |Φ-⟩ = (|00⟩ - |11⟩)/√2.

    Maximally entangled state.
    """
    return TwoQubitSystem([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)])


def bell_psi_plus() -> TwoQubitSystem:
    """
    Create Bell state |Ψ+⟩ = (|01⟩ + |10⟩)/√2.

    Maximally entangled state.
    """
    return TwoQubitSystem([0, 1/np.sqrt(2), 1/np.sqrt(2), 0])


def bell_psi_minus() -> TwoQubitSystem:
    """
    Create Bell state |Ψ-⟩ = (|01⟩ - |10⟩)/√2.

    Maximally entangled state (singlet).
    """
    return TwoQubitSystem([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
