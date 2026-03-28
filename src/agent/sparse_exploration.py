"""Sparse-code exploration bonus for Dyna-SAC.

Uses the topk sparse activation support (set of active atom indices) as a
natural discrete hash for count-based exploration. Novel activation patterns
indicate unexplored dynamics modes, providing intrinsic reward.

This unifies world model and exploration: the same sparse encoder that
predicts transitions also drives exploration — zero extra parameters.
"""

import torch


class SparseCodeExploration:
    """Count-based exploration using sparse code support patterns.

    The topk-sparse alpha vector selects k atoms from K total.
    The support (set of active indices) acts as a state hash.
    Rarely-seen supports get higher exploration bonus.

    Args:
        eta: Exploration bonus scale.
        decay: Count decay factor (0=no decay, <1=forget old counts).
    """

    def __init__(self, eta: float = 0.1, decay: float = 0.0) -> None:
        self.eta = eta
        self.decay = decay
        self._counts: dict[tuple[int, ...], int] = {}

    def compute_bonus(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute exploration bonus from sparse codes.

        Args:
            alpha: Sparse codes, shape (batch, K).

        Returns:
            Intrinsic reward bonus, shape (batch,).
        """
        bonuses = []
        for i in range(alpha.shape[0]):
            # Extract support: indices of nonzero entries
            support = tuple(
                sorted(torch.nonzero(alpha[i].abs() > 1e-6, as_tuple=True)[0].tolist())
            )

            # Count-based bonus: 1/sqrt(N)
            count = self._counts.get(support, 0)
            self._counts[support] = count + 1
            bonus = self.eta / (count + 1) ** 0.5
            bonuses.append(bonus)

        return torch.tensor(bonuses, device=alpha.device, dtype=alpha.dtype)

    def apply_decay(self) -> None:
        """Decay all counts (call at episode boundary)."""
        if self.decay > 0:
            for key in self._counts:
                self._counts[key] = int(self._counts[key] * (1 - self.decay))

    @property
    def n_unique_patterns(self) -> int:
        """Number of unique activation patterns seen."""
        return len(self._counts)

    @property
    def stats(self) -> dict[str, float]:
        """Exploration statistics."""
        if not self._counts:
            return {"n_patterns": 0, "mean_count": 0, "max_count": 0}
        counts = list(self._counts.values())
        return {
            "n_patterns": float(len(counts)),
            "mean_count": sum(counts) / len(counts),
            "max_count": float(max(counts)),
        }
