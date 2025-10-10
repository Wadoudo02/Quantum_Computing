"""
Qubit Class Implementation
==========================

A qubit is a two-level quantum system represented as a complex vector in C^2.

This module implements the fundamental qubit object with:
- State vector representation
- Normalisation
- Measurement simulation
- Bloch sphere coordinates
"""

import numpy as np
from typing import Tuple, Optional


class Qubit:
    """
    A quantum bit (qubit) represented as a state vector.
    
    The qubit state is represented as |ψ⟩ = α|0⟩ + β|1⟩ where:
    - α, β are complex amplitudes
    - |α|² + |β|² = 1 (normalisation condition)
    
    Attributes
    ----------
    state : np.ndarray
        Complex vector of shape (2,) representing the quantum state
    
    Examples
    --------
    >>> # Create |0⟩ state
    >>> q = Qubit([1, 0])
    >>> print(q)
    |0⟩
    
    >>> # Create superposition |+⟩ = (|0⟩ + |1⟩)/√2
    >>> q = Qubit([1/np.sqrt(2), 1/np.sqrt(2)])
    >>> print(q)
    0.71|0⟩ + 0.71|1⟩
    """
    
    def __init__(self, state: list | np.ndarray, normalize: bool = True):
        """
        Initialise a qubit with a given state.
        
        Parameters
        ----------
        state : list or np.ndarray
            Two-element array [α, β] representing α|0⟩ + β|1⟩
        normalize : bool, optional
            Whether to normalise the state (default: True)
        
        Raises
        ------
        ValueError
            If state is not two-dimensional
        """
        self.state = np.array(state, dtype=complex)
        
        if len(self.state) != 2:
            raise ValueError("Qubit state must be 2-dimensional")
        
        if normalize:
            self._normalize()
    
    def _normalize(self) -> None:
        """Normalise the state vector to unit length."""
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state = self.state / norm
    
    @property
    def alpha(self) -> complex:
        """Amplitude of |0⟩ component."""
        return self.state[0]
    
    @property
    def beta(self) -> complex:
        """Amplitude of |1⟩ component."""
        return self.state[1]
    
    def prob_0(self) -> float:
        """
        Probability of measuring |0⟩.
        
        Returns
        -------
        float
            P(0) = |α|²
        """
        return np.abs(self.alpha) ** 2
    
    def prob_1(self) -> float:
        """
        Probability of measuring |1⟩.
        
        Returns
        -------
        float
            P(1) = |β|²
        """
        return np.abs(self.beta) ** 2
    
    def measure(self, shots: int = 1) -> np.ndarray:
        """
        Simulate measurement in computational basis {|0⟩, |1⟩}.
        
        Uses Born rule: P(outcome) = |⟨outcome|ψ⟩|²
        
        Parameters
        ----------
        shots : int, optional
            Number of measurements to simulate (default: 1)
        
        Returns
        -------
        np.ndarray
            Array of measurement outcomes (0s and 1s)
        
        Examples
        --------
        >>> q = Qubit([1, 0])  # |0⟩ state
        >>> q.measure(10)
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        >>> q = Qubit([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩ state
        >>> q.measure(100)  # Should give roughly 50% 0s and 50% 1s
        """
        probabilities = [self.prob_0(), self.prob_1()]
        outcomes = np.random.choice([0, 1], size=shots, p=probabilities)
        return outcomes
    
    def bloch_coordinates(self) -> Tuple[float, float, float]:
        """
        Calculate Bloch sphere coordinates (x, y, z).
        
        For a qubit |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩:
        - x = sin(θ)cos(φ)
        - y = sin(θ)sin(φ)  
        - z = cos(θ)
        
        Returns
        -------
        tuple of float
            (x, y, z) coordinates on the unit sphere
        
        Notes
        -----
        The Bloch sphere is a geometrical representation where:
        - North pole (0, 0, 1) represents |0⟩
        - South pole (0, 0, -1) represents |1⟩
        - Equator represents equal superpositions
        """
        alpha = self.alpha
        beta = self.beta
        
        # Calculate Bloch vector components
        # From ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ expectation values
        x = 2 * np.real(np.conj(alpha) * beta)
        y = 2 * np.imag(np.conj(alpha) * beta)
        z = np.abs(alpha) ** 2 - np.abs(beta) ** 2
        
        return (x, y, z)
    
    def bloch_angles(self) -> Tuple[float, float]:
        """
        Calculate Bloch sphere angles (θ, φ) in radians.
        
        Returns
        -------
        tuple of float
            (theta, phi) where:
            - theta ∈ [0, π] is the polar angle
            - phi ∈ [0, 2π) is the azimuthal angle
        """
        x, y, z = self.bloch_coordinates()
        
        # Calculate angles from Cartesian coordinates
        theta = np.arccos(np.clip(z, -1, 1))  # Clip to handle numerical errors
        phi = np.arctan2(y, x) % (2 * np.pi)
        
        return (theta, phi)
    
    def copy(self) -> 'Qubit':
        """
        Create a copy of this qubit.
        
        Returns
        -------
        Qubit
            A new Qubit object with the same state
        """
        return Qubit(self.state.copy(), normalize=False)
    
    def is_normalized(self, tolerance: float = 1e-10) -> bool:
        """
        Check if the state is normalised.
        
        Parameters
        ----------
        tolerance : float, optional
            Numerical tolerance for normalisation check
        
        Returns
        -------
        bool
            True if |α|² + |β|² ≈ 1
        """
        norm_squared = self.prob_0() + self.prob_1()
        return np.abs(norm_squared - 1.0) < tolerance
    
    def __repr__(self) -> str:
        """Programmer-friendly representation."""
        return f"Qubit(state={self.state})"
    
    def __str__(self) -> str:
        """
        Human-readable representation in Dirac notation.
        
        Returns
        -------
        str
            String like "0.71|0⟩ + 0.71|1⟩" or "|0⟩" for basis states
        """
        alpha = self.alpha
        beta = self.beta
        
        # Special cases for basis states
        if np.abs(alpha - 1) < 1e-10 and np.abs(beta) < 1e-10:
            return "|0⟩"
        if np.abs(beta - 1) < 1e-10 and np.abs(alpha) < 1e-10:
            return "|1⟩"
        
        # General case
        parts = []
        
        if np.abs(alpha) > 1e-10:
            if np.abs(np.imag(alpha)) < 1e-10:
                parts.append(f"{np.real(alpha):.2f}|0⟩")
            else:
                parts.append(f"({alpha:.2f})|0⟩")
        
        if np.abs(beta) > 1e-10:
            if np.real(beta) >= 0 and parts:
                connector = " + "
            else:
                connector = " " if not parts else " "
            
            if np.abs(np.imag(beta)) < 1e-10:
                parts.append(f"{connector}{np.real(beta):.2f}|1⟩")
            else:
                parts.append(f"{connector}({beta:.2f})|1⟩")
        
        return "".join(parts) if parts else "0|0⟩"
    
    def __eq__(self, other: 'Qubit') -> bool:
        """
        Check if two qubits have the same state (up to global phase).
        
        Note: Two states that differ only by a global phase e^(iθ) 
        represent the same physical state.
        """
        if not isinstance(other, Qubit):
            return False
        
        # Check if states are equal up to a global phase
        # |ψ⟩ and e^(iθ)|ψ⟩ are equivalent
        inner_product = np.vdot(self.state, other.state)
        return np.abs(np.abs(inner_product) - 1.0) < 1e-10


# Convenience functions for common states
def ket_0() -> Qubit:
    """Create the |0⟩ basis state."""
    return Qubit([1, 0])


def ket_1() -> Qubit:
    """Create the |1⟩ basis state."""
    return Qubit([0, 1])


def ket_plus() -> Qubit:
    """Create the |+⟩ state = (|0⟩ + |1⟩)/√2."""
    return Qubit([1/np.sqrt(2), 1/np.sqrt(2)])


def ket_minus() -> Qubit:
    """Create the |−⟩ state = (|0⟩ − |1⟩)/√2."""
    return Qubit([1/np.sqrt(2), -1/np.sqrt(2)])


def random_qubit() -> Qubit:
    """
    Create a random qubit state uniformly distributed on the Bloch sphere.
    
    Uses the method of generating random angles θ and φ.
    """
    theta = np.arccos(2 * np.random.random() - 1)  # Uniform on sphere
    phi = 2 * np.pi * np.random.random()
    
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    
    return Qubit([alpha, beta], normalize=False)