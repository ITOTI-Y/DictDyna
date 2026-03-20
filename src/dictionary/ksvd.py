"""K-SVD dictionary learning implementation."""

import numpy as np
import torch
from loguru import logger


class KSVDDictionary:
    """K-SVD dictionary learning with OMP sparse coding.

    Learns an overcomplete dictionary D such that data X can be
    represented as X ≈ D @ A, where A is sparse.

    Args:
        n_atoms: Number of dictionary atoms K.
        n_nonzero: Max nonzero coefficients per sample in OMP.
        max_iter: Number of K-SVD iterations.
        tol: Convergence tolerance on reconstruction error.
    """

    def __init__(
        self,
        n_atoms: int = 128,
        n_nonzero: int = 10,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> None:
        self.n_atoms = n_atoms
        self.n_nonzero = n_nonzero
        self.max_iter = max_iter
        self.tol = tol
        self.dictionary: np.ndarray | None = None  # shape (d, K)

    def fit(self, data: np.ndarray) -> "KSVDDictionary":
        """Fit dictionary to data using K-SVD.

        Args:
            data: Training data, shape (N, d).

        Returns:
            self
        """
        n_samples, _d = data.shape
        # Initialize dictionary with random data samples
        indices = np.random.choice(n_samples, self.n_atoms, replace=False)
        self.dictionary = data[indices].T.copy()  # (d, K)
        self._normalize_atoms()

        prev_error = float("inf")
        for iteration in range(self.max_iter):
            # Sparse coding step: OMP for each sample
            codes = self._omp_batch(data)  # (N, K)

            # Dictionary update step: update each atom
            for k in range(self.n_atoms):
                # Find samples that use atom k
                mask = codes[:, k] != 0
                if not np.any(mask):
                    continue

                # Compute residual without atom k
                codes_k = codes[mask].copy()
                codes_k[:, k] = 0
                residual = data[mask] - codes_k @ self.dictionary.T  # (n_use, d)

                # SVD on residual
                u, s, vt = np.linalg.svd(residual.T, full_matrices=False)
                self.dictionary[:, k] = u[:, 0]
                codes[mask, k] = s[0] * vt[0, :]

            self._normalize_atoms()

            # Check convergence
            reconstruction = codes @ self.dictionary.T
            error = np.mean((data - reconstruction) ** 2)
            logger.debug(f"K-SVD iter {iteration}: MSE={error:.6f}")
            if abs(prev_error - error) < self.tol:
                logger.info(f"K-SVD converged at iteration {iteration}")
                break
            prev_error = error

        return self

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Sparse encode data using OMP.

        Args:
            data: Input data, shape (N, d).

        Returns:
            Sparse codes, shape (N, K).
        """
        return self._omp_batch(data)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """Reconstruct data through dictionary.

        Args:
            data: Input data, shape (N, d).

        Returns:
            Reconstructed data, shape (N, d).
        """
        codes = self.encode(data)
        return codes @ self.dictionary.T  # ty: ignore[unresolved-attribute]

    def to_torch(self, device: str = "cpu") -> torch.Tensor:
        """Convert dictionary to PyTorch tensor.

        Returns:
            Dictionary tensor, shape (d, K).
        """
        assert self.dictionary is not None, "Dictionary not fitted"
        return torch.tensor(self.dictionary, dtype=torch.float32, device=device)

    @classmethod
    def from_torch(cls, tensor: torch.Tensor) -> "KSVDDictionary":
        """Create KSVDDictionary from a PyTorch tensor."""
        obj = cls(n_atoms=tensor.shape[1])
        obj.dictionary = tensor.detach().cpu().numpy()
        return obj

    def save(self, path: str) -> None:
        """Save dictionary to file."""
        assert self.dictionary is not None, "Dictionary not fitted"
        np.save(path, self.dictionary)

    def load(self, path: str) -> "KSVDDictionary":
        """Load dictionary from file."""
        self.dictionary = np.load(path)
        self.n_atoms = self.dictionary.shape[1]
        return self

    def _normalize_atoms(self) -> None:
        """Normalize dictionary columns to unit L2 norm."""
        norms = np.linalg.norm(self.dictionary, axis=0, keepdims=True)  # ty: ignore[no-matching-overload]
        norms = np.maximum(norms, 1e-10)
        self.dictionary /= norms

    def _omp_batch(self, data: np.ndarray) -> np.ndarray:
        """Orthogonal Matching Pursuit for batch of samples.

        Args:
            data: Input data, shape (N, d).

        Returns:
            Sparse codes, shape (N, K).
        """
        assert self.dictionary is not None, "Dictionary not fitted"
        n_samples = data.shape[0]
        codes = np.zeros((n_samples, self.n_atoms))

        for i in range(n_samples):
            codes[i] = self._omp_single(data[i])

        return codes

    def _omp_single(self, x: np.ndarray) -> np.ndarray:
        """OMP for a single sample.

        Args:
            x: Single data point, shape (d,).

        Returns:
            Sparse code, shape (K,).
        """
        residual = x.copy()
        support: list[int] = []
        code = np.zeros(self.n_atoms)

        for _ in range(self.n_nonzero):
            # Find atom with largest correlation to residual
            correlations = self.dictionary.T @ residual  # ty: ignore[unresolved-attribute]
            k = int(np.argmax(np.abs(correlations)))

            if k in support:
                break
            support.append(k)

            # Solve least squares on support set
            d_support = self.dictionary[:, support]  # ty: ignore[not-subscriptable]
            coeffs, _, _, _ = np.linalg.lstsq(d_support, x, rcond=None)

            # Update residual
            residual = x - d_support @ coeffs

            if np.linalg.norm(residual) < 1e-10:
                break

        # Fill in coefficients
        if support:
            d_support = self.dictionary[:, support]  # ty: ignore[not-subscriptable]
            coeffs, _, _, _ = np.linalg.lstsq(d_support, x, rcond=None)
            for j, k in enumerate(support):
                code[k] = coeffs[j]

        return code
