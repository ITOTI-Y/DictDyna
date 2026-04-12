"""Universal observation encoder for heterogeneous Sinergym buildings.

Encodes raw observations of ANY dimension into a fixed-size embedding
by splitting obs into physical categories (time, weather, zone temps, etc.),
processing each with a dedicated fixed-size MLP module, and concatenating.

Within each category, variable-count inputs are handled via pad + mask:
same-category variables share the same MLP weights (e.g., all zone temps
go through the same ZONE_TEMP module regardless of how many zones exist).

Architecture:
    raw_obs (d_i) → split by category → per-category MLP(pad_to → embed)
                  → concat → d_embed (128 fixed)

The encoder is building-agnostic: ONE instance handles all buildings.
Each building provides a CategoryMapping (auto-generated from env var names)
that specifies which raw dims belong to which category.
"""

import torch
import torch.nn as nn

from src.obs_config_universal import (
    CATEGORIES,
    CATEGORY_EMBED_DIM,
    CATEGORY_PAD,
    CategoryMapping,
)


class CategoryModule(nn.Module):
    """MLP module for one observation category.

    Processes padded + masked inputs of a fixed max size,
    producing a fixed-size embedding.
    """

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with masking.

        Args:
            x: Padded category values, shape (batch, pad_to).
            mask: Valid-dim mask, shape (pad_to,). 1=valid, 0=padding.

        Returns:
            Category embedding, shape (batch, embed_dim).
        """
        return self.net(x * mask)


class UniversalObsEncoder(nn.Module):
    """Building-agnostic observation encoder.

    Splits raw observations into 8 physical categories and processes each
    with a dedicated CategoryModule. The output is a fixed 128-dim embedding
    regardless of the building's raw obs_dim.

    Usage:
        encoder = UniversalObsEncoder()
        mapping = build_category_mapping("office_hot", env.observation_variables)
        embed = encoder(raw_obs_tensor, mapping)
    """

    def __init__(self, embed_dim: int = CATEGORY_EMBED_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.total_embed_dim = embed_dim * len(CATEGORIES)

        self.modules_dict = nn.ModuleDict(
            {cat: CategoryModule(CATEGORY_PAD[cat], embed_dim) for cat in CATEGORIES}
        )

    def forward(
        self,
        raw_obs: torch.Tensor,
        mapping: CategoryMapping,
    ) -> torch.Tensor:
        """Encode raw observation to fixed-size embedding.

        Args:
            raw_obs: Raw observation tensor, shape (batch, obs_dim).
            mapping: CategoryMapping for the current building.

        Returns:
            Embedding tensor, shape (batch, total_embed_dim).
        """
        device = raw_obs.device
        embeddings = []

        for cat in CATEGORIES:
            indices = mapping.category_indices[cat]
            pad_to = CATEGORY_PAD[cat]
            mask = torch.tensor(
                mapping.category_masks[cat], dtype=torch.float32, device=device
            )

            if len(indices) == 0:
                padded = torch.zeros(
                    raw_obs.shape[0], pad_to, dtype=torch.float32, device=device
                )
            else:
                values = raw_obs[:, list(indices)]
                if values.shape[1] < pad_to:
                    padding = torch.zeros(
                        raw_obs.shape[0],
                        pad_to - values.shape[1],
                        dtype=torch.float32,
                        device=device,
                    )
                    padded = torch.cat([values, padding], dim=1)
                else:
                    padded = values

            cat_embed = self.modules_dict[cat](padded, mask)
            embeddings.append(cat_embed)

        return torch.cat(embeddings, dim=1)

    def encode_numpy(
        self,
        raw_obs_np: "np.ndarray",  # noqa: F821
        mapping: CategoryMapping,
    ) -> torch.Tensor:
        """Convenience: encode numpy obs to tensor embedding."""
        raw_t = torch.tensor(raw_obs_np, dtype=torch.float32)
        if raw_t.dim() == 1:
            raw_t = raw_t.unsqueeze(0)
        with torch.no_grad():
            return self(raw_t, mapping)
