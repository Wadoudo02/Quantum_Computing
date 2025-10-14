"""
Bell States and Two-Qubit Systems
==================================

Implementation of Bell states and two-qubit quantum systems for exploring
entanglement and non-local correlations.

Bell states are the four maximally entangled two-qubit states:
- |Φ+⟩ = (|00⟩ + |11⟩)/√2
- |Φ-⟩ = (|00⟩ - |11⟩)/√2
- |Ψ+⟩ = (|01⟩ + |10⟩)/√2
- |Ψ-⟩ = (|01⟩ - |10⟩)/√2

Reference: Imperial College notes on entanglement and Bell states
"""

import numpy as np
from typing import Tuple, Optional, List
import sys
from pathlib import Path

# Import from Phase 1
sys.path.insert(0, str(Path(__file__).parent.parent))
from phase1_qubits.qubit import Qubit


class BellState:
    """
    Represents a two-qubit quantum system, with special support for Bell states.

    The state is represented in the computational basis:
    |ψ⟩ = c₀₀|00⟩ + c₀₁|01⟩ + c₁₀|10⟩ + c₁₁|11⟩

    Attributes
    ----------
    state : np.ndarray
        Complex vector of shape (4,) in basis [|00⟩, |01⟩, |10⟩, |11⟩]
    name : str, optional
        Name of the state (e.g., "Φ+", "Ψ-")
    """

    def __init__(self, state: np.ndarray | list, name: Optional[str] = None, normalize: bool = True):
        """
        Initialize a two-qubit system.

        Parameters
        ----------
        state : np.ndarray or list
            4-element array [c₀₀, c₀₁, c₁₀, c₁₁]
        name : str, optional
            Name for the state
        normalize : bool, optional
            Whether to normalize (default: True)
        """
        self.state = np.array(state, dtype=complex)
        self.name = name

        if len(self.state) != 4:
            raise ValueError("Two-qubit state must be 4-dimensional")

        if normalize:
            self._normalize()

    def _normalize(self) -> None:
        """Normalize the state vector to unit length."""
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state = self.state / norm

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
        shots : int
            Number of measurements

        Returns
        -------
        np.ndarray
            Array of shape (shots, 2) with measurement outcomes
            Each row is [qubit1_outcome, qubit2_outcome]
        """
        probabilities = np.abs(self.state) ** 2
        outcomes_flat = np.random.choice(4, size=shots, p=probabilities)

        # Convert to binary: 0→[0,0], 1→[0,1], 2→[1,0], 3→[1,1]
        outcomes = np.zeros((shots, 2), dtype=int)
        outcomes[:, 0] = outcomes_flat // 2
        outcomes[:, 1] = outcomes_flat % 2

        return outcomes

    def measure_in_basis(self, basis1: np.ndarray, basis2: np.ndarray, shots: int = 1) -> np.ndarray:
        """
        Measure qubits in arbitrary single-qubit bases.

        This is crucial for Bell's inequality tests where we measure
        in different bases (e.g., rotated bases on Bloch sphere).

        Parameters
        ----------
        basis1 : np.ndarray
            2x2 unitary matrix defining measurement basis for qubit 1
        basis2 : np.ndarray
            2x2 unitary matrix defining measurement basis for qubit 2
        shots : int
            Number of measurements

        Returns
        -------
        np.ndarray
            Array of shape (shots, 2) with measurement outcomes in new basis
        """
        # Transform to measurement basis
        measurement_operator = np.kron(basis1, basis2)
        rotated_state = measurement_operator.conj().T @ self.state

        # Measure in computational basis of rotated state
        probabilities = np.abs(rotated_state) ** 2
        outcomes_flat = np.random.choice(4, size=shots, p=probabilities)

        outcomes = np.zeros((shots, 2), dtype=int)
        outcomes[:, 0] = outcomes_flat // 2
        outcomes[:, 1] = outcomes_flat % 2

        return outcomes

    def density_matrix(self) -> np.ndarray:
        """
        Compute density matrix representation.

        For pure state: ρ = |ψ⟩⟨ψ|

        Returns
        -------
        np.ndarray
            4x4 density matrix
        """
        return np.outer(self.state, self.state.conj())

    def reduced_density_matrix(self, subsystem: int = 0) -> np.ndarray:
        """
        Compute reduced density matrix by tracing out one qubit.

        This is the partial trace operation that gives us the state
        of one qubit when we ignore the other.

        Parameters
        ----------
        subsystem : int
            Which qubit to keep (0 or 1)

        Returns
        -------
        np.ndarray
            2x2 reduced density matrix
        """
        rho = self.density_matrix()

        if subsystem == 0:
            # Trace out qubit 2
            rho_A = rho[[0,2], :][:, [0,2]] + rho[[1,3], :][:, [1,3]]
        else:
            # Trace out qubit 1
            rho_B = rho[[0,1], :][:, [0,1]] + rho[[2,3], :][:, [2,3]]
            return rho_B

        return rho_A

    def is_entangled(self, tolerance: float = 1e-10) -> bool:
        """
        Check if the state is entangled using Schmidt decomposition.

        A state is entangled if it has Schmidt rank > 1, meaning it
        cannot be written as a product state |ψ⟩ = |a⟩ ⊗ |b⟩.

        Parameters
        ----------
        tolerance : float
            Numerical tolerance for zero eigenvalues

        Returns
        -------
        bool
            True if entangled, False if separable
        """
        # Reshape state into 2x2 matrix and compute SVD
        psi_matrix = self.state.reshape(2, 2)
        singular_values = np.linalg.svd(psi_matrix, compute_uv=False)

        # Count non-zero singular values (Schmidt rank)
        schmidt_rank = np.sum(singular_values > tolerance)

        return schmidt_rank > 1

    def schmidt_decomposition(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Schmidt decomposition of the two-qubit state.

        Every bipartite pure state can be written as:
        |ψ⟩ = Σᵢ √λᵢ |iₐ⟩|iᵦ⟩

        where λᵢ are the Schmidt coefficients (eigenvalues).

        Returns
        -------
        coefficients : np.ndarray
            Schmidt coefficients (√λᵢ)
        basis_A : np.ndarray
            Schmidt basis for subsystem A
        basis_B : np.ndarray
            Schmidt basis for subsystem B
        """
        psi_matrix = self.state.reshape(2, 2)
        U, singular_values, Vh = np.linalg.svd(psi_matrix)

        return singular_values, U, Vh.conj().T

    def von_neumann_entropy(self) -> float:
        """
        Calculate von Neumann entropy of the reduced density matrix.

        For a pure bipartite state, this quantifies entanglement:
        S(ρₐ) = -Tr(ρₐ log₂ ρₐ)

        Returns
        -------
        float
            Entropy in bits (0 = separable, 1 = maximally entangled for 2 qubits)
        """
        rho_A = self.reduced_density_matrix(0)
        eigenvalues = np.linalg.eigvalsh(rho_A)

        # Filter out zero eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        # S = -Σᵢ λᵢ log₂(λᵢ)
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

        return entropy

    def __str__(self) -> str:
        """String representation showing amplitudes."""
        terms = []
        basis_labels = ['00', '01', '10', '11']

        for i, (coeff, label) in enumerate(zip(self.state, basis_labels)):
            if np.abs(coeff) > 1e-10:
                if np.abs(coeff.imag) < 1e-10:
                    # Real coefficient
                    if np.abs(coeff.real - 1.0) < 1e-10:
                        terms.append(f"|{label}⟩")
                    elif np.abs(coeff.real + 1.0) < 1e-10:
                        terms.append(f"-|{label}⟩")
                    else:
                        terms.append(f"{coeff.real:.3f}|{label}⟩")
                else:
                    # Complex coefficient
                    terms.append(f"({coeff.real:.3f}{coeff.imag:+.3f}j)|{label}⟩")

        if not terms:
            return "0"

        result = " + ".join(terms).replace("+ -", "- ")

        if self.name:
            return f"|{self.name}⟩ = {result}"
        return result

    def __repr__(self) -> str:
        """Programmer-friendly representation."""
        return f"BellState(state={self.state}, name={self.name})"

    def copy(self) -> 'BellState':
        """Create a copy of this state."""
        return BellState(self.state.copy(), name=self.name, normalize=False)


# ============================================================================
# Bell State Constructors
# ============================================================================

def bell_phi_plus() -> BellState:
    """
    Create the |Φ+⟩ Bell state: (|00⟩ + |11⟩)/√2

    This is a maximally entangled state where measuring one qubit
    immediately determines the other's state (both give same result).

    Returns
    -------
    BellState
        The |Φ+⟩ state
    """
    state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    return BellState(state, name="Φ+", normalize=False)


def bell_phi_minus() -> BellState:
    """
    Create the |Φ-⟩ Bell state: (|00⟩ - |11⟩)/√2

    Returns
    -------
    BellState
        The |Φ-⟩ state
    """
    state = np.array([1, 0, 0, -1]) / np.sqrt(2)
    return BellState(state, name="Φ-", normalize=False)


def bell_psi_plus() -> BellState:
    """
    Create the |Ψ+⟩ Bell state: (|01⟩ + |10⟩)/√2

    Returns
    -------
    BellState
        The |Ψ+⟩ state
    """
    state = np.array([0, 1, 1, 0]) / np.sqrt(2)
    return BellState(state, name="Ψ+", normalize=False)


def bell_psi_minus() -> BellState:
    """
    Create the |Ψ-⟩ Bell state: (|01⟩ - |10⟩)/√2

    Returns
    -------
    BellState
        The |Ψ-⟩ state
    """
    state = np.array([0, 1, -1, 0]) / np.sqrt(2)
    return BellState(state, name="Ψ-", normalize=False)


def create_bell_state(state_type: str) -> BellState:
    """
    Create a Bell state by name.

    Parameters
    ----------
    state_type : str
        One of: "phi_plus", "phi_minus", "psi_plus", "psi_minus"
        or "Φ+", "Φ-", "Ψ+", "Ψ-"

    Returns
    -------
    BellState
        The requested Bell state
    """
    state_map = {
        "phi_plus": bell_phi_plus,
        "phi_minus": bell_phi_minus,
        "psi_plus": bell_psi_plus,
        "psi_minus": bell_psi_minus,
        "Φ+": bell_phi_plus,
        "Φ-": bell_phi_minus,
        "Ψ+": bell_psi_plus,
        "Ψ-": bell_psi_minus,
    }

    if state_type not in state_map:
        raise ValueError(f"Unknown Bell state: {state_type}")

    return state_map[state_type]()


def tensor_product(qubit1: Qubit, qubit2: Qubit) -> BellState:
    """
    Create a two-qubit system from tensor product of single qubits.

    |ψ⟩ = |a⟩ ⊗ |b⟩

    Parameters
    ----------
    qubit1 : Qubit
        First qubit
    qubit2 : Qubit
        Second qubit

    Returns
    -------
    BellState
        Product state (generally not entangled)
    """
    state = np.kron(qubit1.state, qubit2.state)
    return BellState(state, normalize=False)
