"""Online dictionary learning using scikit-learn."""

import numpy as np
import torch
from sklearn.decomposition import MiniBatchDictionaryLearning


class OnlineDictionaryLearner:
    """Online dictionary learning wrapper around scikit-learn.

    Supports incremental updates for streaming data.

    Args:
        n_atoms: Number of dictionary atoms K.
        alpha: Sparsity controlling parameter (L1 penalty).
        batch_size: Mini-batch size for online updates.
        n_iter: Number of iterations per partial_fit call.
    """

    def __init__(
        self,
        n_atoms: int = 128,
        alpha: float = 1.0,
        batch_size: int = 256,
        n_iter: int = 100,
    ) -> None:
        self.n_atoms = n_atoms
        self.model = MiniBatchDictionaryLearning(
            n_components=n_atoms,
            alpha=alpha,
            batch_size=batch_size,
            max_iter=n_iter,
            transform_algorithm="lasso_lars",
        )
        self._fitted = False

    def fit(self, data: np.ndarray) -> "OnlineDictionaryLearner":
        """Full fit on data.

        Args:
            data: Training data, shape (N, d).

        Returns:
            self
        """
        self.model.fit(data)
        self._fitted = True
        return self

    def partial_fit(self, data: np.ndarray) -> "OnlineDictionaryLearner":
        """Incremental update with new data.

        Args:
            data: New data batch, shape (N, d).

        Returns:
            self
        """
        self.model.partial_fit(data)
        self._fitted = True
        return self

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Sparse encode data.

        Args:
            data: Input data, shape (N, d).

        Returns:
            Sparse codes, shape (N, K).
        """
        return self.model.transform(data)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        """Reconstruct data through dictionary."""
        codes = self.encode(data)
        return codes @ self.dictionary

    @property
    def dictionary(self) -> np.ndarray:
        """Get dictionary matrix, shape (K, d).

        Note: scikit-learn stores dictionary as (K, d), transposed from our convention.
        """
        return self.model.components_

    def to_torch(self, device: str = "cpu") -> torch.Tensor:
        """Convert to PyTorch tensor, shape (d, K) (our convention)."""
        return torch.tensor(self.dictionary.T, dtype=torch.float32, device=device)
