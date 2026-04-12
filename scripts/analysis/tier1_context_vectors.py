"""T1.4: Context vector analysis.

Tests whether the context encoder learns building-specific fingerprints,
and explains why this is irrelevant to pure zero-shot (actor doesn't use z).

Approach:
    1. Train a ContextDynamicsModel on offline source data (hot+mixed)
    2. Infer context vectors z for all 3 buildings using sliding windows
    3. Visualize: t-SNE of z colored by building + cosine similarity matrix

Outputs:
    figures/context_tsne.png      — t-SNE of context vectors by building
    figures/context_similarity.png — Cosine similarity matrix
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from sklearn.manifold import TSNE

from _share import (
    BUILDING_COLORS,
    BUILDING_LABELS,
    BUILDINGS,
    FIG_DIR,
    SOURCE_BUILDINGS,
    ensure_dirs,
    load_building_transitions,
    load_pretrained_dict,
    normalize_states_fast,
    setup_figure,
)

from src.schemas import TrainSchema
from src.world_model.factory import build_world_model

STATE_DIM = 17
ACTION_DIM = 2
CONTEXT_WINDOW = 10
N_TRAIN_EPOCHS = 30
BATCH_SIZE = 256


def build_transition_windows(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    window_size: int = CONTEXT_WINDOW,
) -> np.ndarray:
    """Build sliding transition windows (s, a, delta_s) for context inference.

    Returns:
        Array of shape (N - window_size, window_size, 2*d + m).
    """
    deltas = next_states - states
    # Concatenate (s, a, delta) per step
    per_step = np.concatenate([states, actions, deltas], axis=-1)  # (N, 2d+m)
    N = len(per_step)
    n_windows = N - window_size
    windows = np.zeros((n_windows, window_size, per_step.shape[1]), dtype=np.float32)
    for i in range(n_windows):
        windows[i] = per_step[i : i + window_size]
    return windows


def main():
    setup_figure()
    ensure_dirs()

    pretrained = load_pretrained_dict()
    obs_mean, obs_std = pretrained["obs_mean"], pretrained["obs_std"]
    dictionary = pretrained["dictionary"]

    # Normalize all buildings' transitions with source stats
    norm_trans: dict[str, dict[str, np.ndarray]] = {}
    for bid in BUILDINGS:
        raw = load_building_transitions(bid)
        norm_trans[bid] = {
            "states": normalize_states_fast(raw["states"], obs_mean, obs_std),
            "actions": raw["actions"],
            "next_states": normalize_states_fast(raw["next_states"], obs_mean, obs_std),
        }

    # === Phase 1: Train ContextDynamicsModel on source offline data ===
    logger.info("Building ContextDynamicsModel for offline training...")
    config = TrainSchema(mode="context", device="cpu")
    model = build_world_model(
        dictionary=dictionary,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        config=config,
        device=torch.device("cpu"),
    )

    # Prepare source training data
    source_s = np.concatenate([norm_trans[b]["states"] for b in SOURCE_BUILDINGS])
    source_a = np.concatenate([norm_trans[b]["actions"] for b in SOURCE_BUILDINGS])
    source_sn = np.concatenate([norm_trans[b]["next_states"] for b in SOURCE_BUILDINGS])

    # Build transition windows for context inference during training
    source_windows = build_transition_windows(source_s, source_a, source_sn)
    logger.info(f"Source training data: {len(source_s)} steps, {len(source_windows)} windows")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(42)

    for epoch in range(N_TRAIN_EPOCHS):
        model.train()
        # Sample batch
        idx = rng.choice(len(source_windows), size=BATCH_SIZE, replace=False)
        # For each sample, use the WINDOW for context and the LAST step for prediction
        windows_t = torch.tensor(source_windows[idx])
        # The prediction target: state at window_end + 1
        target_idx = idx + CONTEXT_WINDOW
        target_idx = np.clip(target_idx, 0, len(source_s) - 1)

        s_t = torch.tensor(source_s[target_idx])
        a_t = torch.tensor(source_a[target_idx])
        sn_t = torch.tensor(source_sn[target_idx])

        # Infer context from window
        ctx = model.infer_context(windows_t)
        ctx_expanded = ctx  # already (batch, context_dim)

        loss, info = model.compute_loss(
            s_t, a_t, sn_t, context=ctx_expanded, sparsity_lambda=0.1
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{N_TRAIN_EPOCHS}: loss={loss.item():.6f}")

    model.eval()

    # === Phase 2: Infer context for all buildings ===
    logger.info("Inferring context vectors for all buildings...")
    n_ctx_samples = 500  # number of context windows per building

    all_contexts: dict[str, np.ndarray] = {}
    for bid in BUILDINGS:
        trans = norm_trans[bid]
        windows = build_transition_windows(
            trans["states"], trans["actions"], trans["next_states"]
        )
        # Sample windows uniformly across the episode
        idx = np.linspace(0, len(windows) - 1, n_ctx_samples, dtype=int)
        windows_t = torch.tensor(windows[idx])

        with torch.no_grad():
            ctx = model.infer_context(windows_t).numpy()
        all_contexts[bid] = ctx
        logger.info(f"  {bid}: context shape={ctx.shape}, "
                     f"mean_norm={np.linalg.norm(ctx, axis=1).mean():.3f}")

    # === Figure 4a: t-SNE of context vectors ===
    all_ctx = np.concatenate([all_contexts[b] for b in BUILDINGS])
    all_labels = np.concatenate(
        [np.full(len(all_contexts[b]), i) for i, b in enumerate(BUILDINGS)]
    )

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    ctx_2d = tsne.fit_transform(all_ctx)

    fig, ax = plt.subplots(figsize=(4, 4))
    for i, bid in enumerate(BUILDINGS):
        mask = all_labels == i
        ax.scatter(
            ctx_2d[mask, 0], ctx_2d[mask, 1],
            c=BUILDING_COLORS[bid],
            label=BUILDING_LABELS[bid],
            s=8, alpha=0.5,
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Context Vectors by Building")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "context_tsne.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {FIG_DIR / 'context_tsne.png'}")

    # === Figure 4b: Cosine similarity matrix ===
    mean_contexts = {bid: all_contexts[bid].mean(axis=0) for bid in BUILDINGS}

    # Pairwise cosine similarity
    labels = [BUILDING_LABELS[b] for b in BUILDINGS]
    n = len(BUILDINGS)
    sim_matrix = np.zeros((n, n))
    for i, bi in enumerate(BUILDINGS):
        for j, bj in enumerate(BUILDINGS):
            vi, vj = mean_contexts[bi], mean_contexts[bj]
            cos = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8)
            sim_matrix[i, j] = cos

    fig2, ax2 = plt.subplots(figsize=(3.5, 3))
    im = ax2.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{sim_matrix[i, j]:.2f}",
                     ha="center", va="center", fontsize=9,
                     color="white" if abs(sim_matrix[i, j]) > 0.5 else "black")
    ax2.set_title("Context Vector Cosine Similarity", fontsize=9)
    fig2.colorbar(im, ax=ax2, shrink=0.8)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "context_similarity.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Saved {FIG_DIR / 'context_similarity.png'}")

    # Summary
    print("\n=== Context Vector Summary ===")
    print("Mean cosine similarity between building contexts:")
    for i, bi in enumerate(BUILDINGS):
        for j, bj in enumerate(BUILDINGS):
            if j > i:
                print(f"  {BUILDING_LABELS[bi]} ↔ {BUILDING_LABELS[bj]}: "
                      f"{sim_matrix[i, j]:.3f}")

    print("\nInterpretation:")
    print("  Context vectors differentiate buildings (if cosine < 1.0)")
    print("  BUT the actor does NOT use context → irrelevant to pure zero-shot")
    print("  WM's role: provide shared representation during SOURCE training,")
    print("  not inference-time adaptation")


if __name__ == "__main__":
    main()
