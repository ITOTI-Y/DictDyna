"""Verify that hot/mixed/cool weather files actually differ significantly.

Before concluding that cross-climate transfer is trivial, we must verify
the climates really differ at the weather/obs level.

Computes outdoor temperature, humidity, and solar radiation distributions
across all 3 buildings' offline data.

Outputs:
    figures/climate_comparison.png  — outdoor weather distributions
    tables/climate_stats.csv       — per-climate summary statistics
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from _share import (
    BUILDING_COLORS,
    BUILDING_LABELS,
    BUILDINGS,
    FIG_DIR,
    TABLE_DIR,
    ensure_dirs,
    load_building_transitions,
    setup_figure,
)

# Raw obs dim indices (from obs_config.py)
DIM_MONTH = 0
DIM_OUTDOOR_TEMP = 3
DIM_OUTDOOR_RH = 4
DIM_WIND_SPD = 5
DIM_DIFFUSE_SOL = 7
DIM_DIRECT_SOL = 8


def main():
    setup_figure()
    ensure_dirs()

    # Load raw states (unnormalized) from each building
    raw_states: dict[str, np.ndarray] = {}
    for bid in BUILDINGS:
        trans = load_building_transitions(bid)
        raw_states[bid] = trans["states"]

    # === Figure: Outdoor weather distributions ===
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    titles = [
        ("Outdoor Temperature (°C)", DIM_OUTDOOR_TEMP),
        ("Outdoor Humidity (%)", DIM_OUTDOOR_RH),
        ("Wind Speed (m/s)", DIM_WIND_SPD),
        ("Direct Solar Radiation (W/m²)", DIM_DIRECT_SOL),
    ]

    for ax, (title, dim) in zip(axes.flat, titles, strict=True):
        for bid in BUILDINGS:
            vals = raw_states[bid][:, dim]
            ax.hist(
                vals, bins=50, alpha=0.4,
                color=BUILDING_COLORS[bid], label=BUILDING_LABELS[bid],
                density=True,
            )
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    fig.suptitle("Weather Distribution Comparison (raw obs values)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "climate_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'climate_comparison.png'}")

    # === Figure: Per-month outdoor temperature ===
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for bid in BUILDINGS:
        obs = raw_states[bid]
        months = obs[:, DIM_MONTH].astype(int)
        outdoor_t = obs[:, DIM_OUTDOOR_TEMP]
        month_means = []
        month_stds = []
        month_x = sorted(set(months))
        for m in month_x:
            mask = months == m
            month_means.append(outdoor_t[mask].mean())
            month_stds.append(outdoor_t[mask].std())
        ax2.errorbar(
            month_x, month_means, yerr=month_stds,
            label=BUILDING_LABELS[bid], color=BUILDING_COLORS[bid],
            marker="o", capsize=3,
        )
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Outdoor Temperature (°C)")
    ax2.legend()
    ax2.set_title("Monthly Outdoor Temperature by Climate")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(
        FIG_DIR / "climate_monthly_temp.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig2)
    print(f"Saved {FIG_DIR / 'climate_monthly_temp.png'}")

    # === Summary stats ===
    rows = []
    var_names = {
        DIM_OUTDOOR_TEMP: "outdoor_temp_C",
        DIM_OUTDOOR_RH: "outdoor_rh_%",
        DIM_WIND_SPD: "wind_speed_m/s",
        DIM_DIFFUSE_SOL: "diffuse_solar_W/m2",
        DIM_DIRECT_SOL: "direct_solar_W/m2",
    }

    for bid in BUILDINGS:
        obs = raw_states[bid]
        for dim, name in var_names.items():
            vals = obs[:, dim]
            rows.append({
                "building": BUILDING_LABELS[bid],
                "variable": name,
                "mean": f"{vals.mean():.2f}",
                "std": f"{vals.std():.2f}",
                "min": f"{vals.min():.2f}",
                "max": f"{vals.max():.2f}",
                "p10": f"{np.percentile(vals, 10):.2f}",
                "p90": f"{np.percentile(vals, 90):.2f}",
            })

    table_path = TABLE_DIR / "climate_stats.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {table_path}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("CLIMATE COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    for dim, name in var_names.items():
        print(f"\n{name}:")
        for bid in BUILDINGS:
            vals = raw_states[bid][:, dim]
            print(
                f"  {BUILDING_LABELS[bid]:>6s}: "
                f"mean={vals.mean():>7.2f} std={vals.std():>6.2f} "
                f"range=[{vals.min():>7.2f}, {vals.max():>7.2f}] "
                f"p10-p90=[{np.percentile(vals, 10):>6.2f}, {np.percentile(vals, 90):>6.2f}]"
            )

    # Annual temp range
    print(f"\n{'=' * 80}")
    print("VERDICT:")
    print(f"{'=' * 80}")
    temps = {bid: raw_states[bid][:, DIM_OUTDOOR_TEMP] for bid in BUILDINGS}
    for bid in BUILDINGS:
        t = temps[bid]
        print(
            f"  {BUILDING_LABELS[bid]:>6s}: annual range = "
            f"[{t.min():.1f}, {t.max():.1f}]°C, mean = {t.mean():.1f}°C"
        )
    pair_diff = {
        "hot vs cool": temps["office_hot"].mean() - temps["office_cool"].mean(),
        "hot vs mixed": temps["office_hot"].mean() - temps["office_mixed"].mean(),
        "mixed vs cool": temps["office_mixed"].mean() - temps["office_cool"].mean(),
    }
    print("\n  Mean annual temperature differences:")
    for pair, diff in pair_diff.items():
        print(f"    {pair:<15s} {diff:>+7.2f}°C")


if __name__ == "__main__":
    main()
