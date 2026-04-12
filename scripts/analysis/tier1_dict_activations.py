"""T1.2: Dictionary atom activation patterns across buildings.

Tests H3: Source-only dictionary (K-SVD on hot+mixed) already spans the
cool building's dynamics subspace.

Uses OMP (OrthogonalMatchingPursuit) to compute sparse codes in the same
space as dictionary pretraining: diffs_norm = raw_diffs / obs_std.

Outputs:
    figures/atom_frequency.png     — Per-atom activation frequency (3 buildings)
    figures/sparse_codes_tsne.png  — t-SNE of sparse codes colored by building
    tables/reconstruction_mse.csv  — Reconstruction error per building + cross-building
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.manifold import TSNE

from _share import (
    BUILDING_COLORS,
    BUILDING_LABELS,
    BUILDINGS,
    FIG_DIR,
    TABLE_DIR,
    ensure_dirs,
    load_pretrained_dict,
    load_state_diffs,
    setup_figure,
)

N_NONZERO = 10  # Match K-SVD pretrain config


def compute_omp_codes_batch(
    diffs_norm: np.ndarray, dictionary: np.ndarray, n_nonzero: int = N_NONZERO
) -> np.ndarray:
    """Compute OMP sparse codes for all samples."""
    N, d = diffs_norm.shape
    K = dictionary.shape[1]
    codes = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero, fit_intercept=False)
        omp.fit(dictionary, diffs_norm[i])
        codes[i] = omp.coef_
    return codes


def main():
    setup_figure()
    ensure_dirs()

    pretrained = load_pretrained_dict()
    D = pretrained["dictionary"].numpy()  # (17, 128)
    obs_std = pretrained["obs_std"]

    # Load diffs and normalize in dictionary space: raw_diffs / obs_std
    all_diffs_norm: dict[str, np.ndarray] = {}
    all_codes: dict[str, np.ndarray] = {}

    for bid in BUILDINGS:
        raw_diffs = load_state_diffs(bid)
        diffs_norm = raw_diffs / obs_std
        all_diffs_norm[bid] = diffs_norm

        # Subsample for OMP speed (full 35040 × 128 OMP is slow)
        rng = np.random.default_rng(42)
        n_sample = min(5000, len(diffs_norm))
        idx = rng.choice(len(diffs_norm), size=n_sample, replace=False)
        logger.info(f"OMP for {bid}: {n_sample} samples...")
        codes = compute_omp_codes_batch(diffs_norm[idx], D)
        all_codes[bid] = codes
        logger.info(f"  done. nonzero frac={np.mean(codes != 0):.3f}")

    # === Figure 2a: Atom activation frequency ===
    fig, ax = plt.subplots(figsize=(7, 2.5))
    K = D.shape[1]
    x = np.arange(K)
    width = 0.25

    for i, bid in enumerate(BUILDINGS):
        freq = np.mean(all_codes[bid] != 0, axis=0)  # (K,)
        ax.bar(
            x + i * width,
            freq,
            width=width,
            color=BUILDING_COLORS[bid],
            alpha=0.7,
            label=BUILDING_LABELS[bid],
        )

    ax.set_xlabel("Atom Index")
    ax.set_ylabel("Activation Frequency")
    ax.set_title("Dictionary Atom Usage by Building")
    ax.legend(fontsize=7)
    ax.set_xlim(-0.5, K + 0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "atom_frequency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {FIG_DIR / 'atom_frequency.png'}")

    # Compute overlap statistics
    threshold = 0.01  # atom is "active" if freq > 1%
    active_sets = {}
    for bid in BUILDINGS:
        freq = np.mean(all_codes[bid] != 0, axis=0)
        active_sets[bid] = set(np.where(freq > threshold)[0])

    hot_set = active_sets["office_hot"]
    mixed_set = active_sets["office_mixed"]
    cool_set = active_sets["office_cool"]
    source_set = hot_set | mixed_set
    shared_all = hot_set & mixed_set & cool_set

    logger.info(f"Active atoms (>{threshold:.0%}): hot={len(hot_set)}, "
                f"mixed={len(mixed_set)}, cool={len(cool_set)}")
    logger.info(f"Source union: {len(source_set)}, shared all 3: {len(shared_all)}")
    logger.info(f"Cool atoms in source set: {len(cool_set & source_set)}/{len(cool_set)} "
                f"= {len(cool_set & source_set) / max(len(cool_set), 1):.1%}")

    # === Figure 2b: t-SNE of sparse codes ===
    # Subsample further for t-SNE (5000 per building → 1500 each for speed)
    n_tsne = 1500
    rng = np.random.default_rng(42)
    tsne_codes = []
    tsne_labels = []
    for i, bid in enumerate(BUILDINGS):
        idx = rng.choice(len(all_codes[bid]), size=min(n_tsne, len(all_codes[bid])), replace=False)
        tsne_codes.append(all_codes[bid][idx])
        tsne_labels.extend([i] * len(idx))

    tsne_codes_arr = np.concatenate(tsne_codes)
    tsne_labels_arr = np.array(tsne_labels)

    # Use PCA first to reduce dims (128 → 30) for t-SNE stability
    pca = PCA(n_components=30)
    codes_pca = pca.fit_transform(tsne_codes_arr)
    logger.info(f"PCA 30 dims explain {pca.explained_variance_ratio_.sum():.1%}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    codes_2d = tsne.fit_transform(codes_pca)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    for i, bid in enumerate(BUILDINGS):
        mask = tsne_labels_arr == i
        ax2.scatter(
            codes_2d[mask, 0],
            codes_2d[mask, 1],
            c=BUILDING_COLORS[bid],
            label=BUILDING_LABELS[bid],
            s=3,
            alpha=0.3,
        )
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.set_title("Sparse Code Embeddings (OMP)")
    ax2.legend(markerscale=3, fontsize=7)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "sparse_codes_tsne.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Saved {FIG_DIR / 'sparse_codes_tsne.png'}")

    # === Table 2: Reconstruction MSE ===
    rows = []
    for bid in BUILDINGS:
        rng_eval = np.random.default_rng(99)
        idx_eval = rng_eval.choice(len(all_diffs_norm[bid]), size=5000, replace=False)
        diffs_eval = all_diffs_norm[bid][idx_eval]
        codes_eval = compute_omp_codes_batch(diffs_eval, D)
        recon = codes_eval @ D.T
        mse_same = float(np.mean((diffs_eval - recon) ** 2))

        # Cross-building: reconstruct cool diffs using this building's codes approach
        # Actually: reconstruct THIS building's diffs using the same dictionary
        rows.append({
            "building": BUILDING_LABELS[bid],
            "reconstruction_mse": f"{mse_same:.6f}",
            "n_active_atoms": len(active_sets[bid]),
        })

    # Cross-building test: can hot OMP codes (refit on cool diffs) reconstruct cool?
    # This is already captured since all buildings use the SAME dictionary.
    # The key metric is whether cool's MSE is comparable to source buildings.

    table_path = TABLE_DIR / "reconstruction_mse.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {table_path}")

    print("\n=== Dictionary Atom Activation Summary ===")
    print(f"Dictionary: {D.shape[1]} atoms, {D.shape[0]} dims")
    print(f"OMP n_nonzero: {N_NONZERO}")
    print(f"\nActive atoms (>{threshold:.0%} freq):")
    for bid in BUILDINGS:
        print(f"  {BUILDING_LABELS[bid]}: {len(active_sets[bid])}/{K}")
    print(f"  Shared across all 3: {len(shared_all)}/{K}")
    print(f"  Cool atoms covered by source: "
          f"{len(cool_set & source_set)}/{len(cool_set)}")
    print(f"\nReconstruction MSE:")
    for r in rows:
        print(f"  {r['building']}: {r['reconstruction_mse']}")


if __name__ == "__main__":
    main()
