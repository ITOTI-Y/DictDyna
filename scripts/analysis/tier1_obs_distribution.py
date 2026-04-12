"""T1.1: Observation distribution comparison across buildings.

Tests H1: After source normalization, cool-building observations fall
within the source distribution → explains why source actor generalizes.

Outputs:
    figures/obs_violin.png        — 17-dim violin plot (hot/mixed/cool)
    figures/obs_pca.png           — PCA 2D scatter colored by building
    tables/obs_wasserstein.csv    — Per-dim Wasserstein distance + coverage
"""

import csv

import numpy as np
import ultraplot as up
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

from _share import (
    BUILDING_COLORS,
    BUILDING_LABELS,
    BUILDINGS,
    FIG_DIR,
    OBS_DIM_NAMES,
    SOURCE_BUILDINGS,
    TABLE_DIR,
    ensure_dirs,
    load_building_transitions,
    load_pretrained_dict,
    normalize_states_fast,
    setup_figure,
)


def main():
    setup_figure()
    ensure_dirs()

    # Load data
    pretrained = load_pretrained_dict()
    obs_mean, obs_std = pretrained["obs_mean"], pretrained["obs_std"]

    norm_data: dict[str, np.ndarray] = {}
    for bid in BUILDINGS:
        trans = load_building_transitions(bid)
        norm_data[bid] = normalize_states_fast(trans["states"], obs_mean, obs_std)

    # Merge source data for comparison
    source_norm = np.concatenate([norm_data[b] for b in SOURCE_BUILDINGS])

    # === Figure 1a: Box plot per dimension ===
    import matplotlib.pyplot as plt

    fig_mpl, axes_mpl = plt.subplots(3, 6, figsize=(10, 6))
    axes_flat = axes_mpl.flat

    for idx in range(17):
        ax = axes_flat[idx]
        bp_data = [norm_data[bid][:, idx] for bid in BUILDINGS]
        bp = ax.boxplot(
            bp_data,
            widths=0.6,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, bid in zip(bp["boxes"], BUILDINGS, strict=True):
            patch.set_facecolor(BUILDING_COLORS[bid])
            patch.set_alpha(0.6)

        ax.set_title(OBS_DIM_NAMES[idx], fontsize=7)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["H", "M", "C"], fontsize=6)
        ax.tick_params(axis="y", labelsize=5)

    # Hide unused subplot (17 dims → 3x6=18 slots)
    axes_flat[17].set_visible(False)

    fig_mpl.suptitle("Normalized Observation Distributions by Building", fontsize=10)
    fig_mpl.tight_layout()
    fig_mpl.savefig(FIG_DIR / "obs_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig_mpl)
    print(f"Saved {FIG_DIR / 'obs_boxplot.png'}")

    # === Figure 1b: PCA scatter ===
    all_norm = np.concatenate([norm_data[b] for b in BUILDINGS])
    all_labels = np.concatenate(
        [np.full(len(norm_data[b]), i) for i, b in enumerate(BUILDINGS)]
    )

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_norm)

    fig2, ax2 = up.subplots(figwidth=3.5)
    # Subsample for clarity (3x35040 points is too dense)
    rng = np.random.default_rng(42)
    n_sample = 2000
    for i, bid in enumerate(BUILDINGS):
        mask = all_labels == i
        idx_sample = rng.choice(mask.sum(), size=min(n_sample, mask.sum()), replace=False)
        ax2.scatter(
            pca_result[mask][idx_sample, 0],
            pca_result[mask][idx_sample, 1],
            c=BUILDING_COLORS[bid],
            label=BUILDING_LABELS[bid],
            s=3,
            alpha=0.3,
        )
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax2.set_title("PCA of Normalized Observations")
    ax2.legend(loc="upper right", markerscale=3)
    fig2.savefig(FIG_DIR / "obs_pca.png")
    print(f"Saved {FIG_DIR / 'obs_pca.png'}")

    # === Table 1: Wasserstein distance + coverage ===
    target_norm = norm_data["office_cool"]
    source_min = source_norm.min(axis=0)
    source_max = source_norm.max(axis=0)

    rows = []
    for idx in range(17):
        wd = wasserstein_distance(source_norm[:, idx], target_norm[:, idx])
        in_range = np.mean(
            (target_norm[:, idx] >= source_min[idx])
            & (target_norm[:, idx] <= source_max[idx])
        )
        rows.append({
            "dim": idx,
            "name": OBS_DIM_NAMES[idx],
            "wasserstein": f"{wd:.4f}",
            "cool_in_source_range": f"{in_range:.1%}",
            "source_mean": f"{source_norm[:, idx].mean():.3f}",
            "cool_mean": f"{target_norm[:, idx].mean():.3f}",
            "source_std": f"{source_norm[:, idx].std():.3f}",
            "cool_std": f"{target_norm[:, idx].std():.3f}",
        })

    table_path = TABLE_DIR / "obs_wasserstein.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {table_path}")

    # Print summary
    print("\n=== Observation Distribution Summary ===")
    print(f"{'Dim':<15s} {'Wasserstein':>12s} {'Cool in Src Range':>18s}")
    print("-" * 48)
    for r in rows:
        print(f"{r['name']:<15s} {r['wasserstein']:>12s} {r['cool_in_source_range']:>18s}")

    coverages = [float(r["cool_in_source_range"].rstrip("%")) / 100 for r in rows]
    print(f"\nMean coverage: {np.mean(coverages):.1%}")
    print(f"Min coverage:  {np.min(coverages):.1%} ({OBS_DIM_NAMES[np.argmin(coverages)]})")


if __name__ == "__main__":
    main()
