"""
Generate publication-readable validation-loss figures from Results/val_losses CSVs.

Outputs 6 single-panel and 3 two-panel figures (PNG + EPS) under Jurnal_plots/readable/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

VAL_DIR = Path("Results/val_losses")
OUT_DIR = Path("Jurnal_plots/readable")

LEGEND_FONTSIZE = 20

PUBLICATION_RCPARAMS = {
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": LEGEND_FONTSIZE,
    "lines.linewidth": 2.5,
}

# Colors from plot_maker_neptune.py / existing readable figures
COLOR_ALL_DATA = "#D60000"
COLOR_90 = "#3AB2EE"
COLOR_95 = "#115D97"
COLOR_98 = "#5A47EB"
COLOR_CHIRP = "#3AB2EE"
COLOR_LANE_CS = "#115D97"
COLOR_SIN_CS = "#5A47EB"
COLOR_LANE_ACC = "#3AB2EE"
COLOR_LANE_SLOW = "#115D97"
COLOR_SIN_ACC = "#610891"
COLOR_SIN_SLOW = "#5A47EB"


def apply_publication_style() -> None:
    plt.rcParams.update(PUBLICATION_RCPARAMS)


def load_series(csv_stem: str) -> pd.Series:
    """Load val_loss column from Results/val_losses/<csv_stem>.csv."""
    path = VAL_DIR / f"{csv_stem}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation loss CSV: {path}")
    df = pd.read_csv(path)
    if "val_loss" not in df.columns:
        raise ValueError(f"Expected 'val_loss' column in {path}")
    return df["val_loss"]


def build_series_map(entries: list[tuple[str, str, str]]) -> dict[str, tuple[pd.Series, str]]:
    """Build {display_label: (series, color)} from (csv_stem, label, color) tuples."""
    result: dict[str, tuple[pd.Series, str]] = {}
    for csv_stem, label, color in entries:
        result[label] = (load_series(csv_stem), color)
    return result


def plot_series_on_ax(ax, series_map: dict[str, tuple[pd.Series, str]]) -> None:
    for label, (series, color) in series_map.items():
        sns.lineplot(
            x=range(len(series)),
            y=series.values,
            label=label,
            color=color,
            ax=ax,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.grid(True)
    ax.legend(prop={"size": LEGEND_FONTSIZE})


def save_figure(fig, out_path: Path) -> None:
    """Save the same figure as PNG and EPS."""
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    eps_path = out_path.with_suffix(".eps")
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"Saved: {eps_path}")


def plot_single(series_map: dict[str, tuple[pd.Series, str]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_series_on_ax(ax, series_map)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_two_panel(
    left_map: dict[str, tuple[pd.Series, str]],
    right_map: dict[str, tuple[pd.Series, str]],
    out_path: Path,
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 7))
    plot_series_on_ax(ax_left, left_map)
    plot_series_on_ax(ax_right, right_map)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def tesla_all_data_stem() -> str:
    return "ae_OG_remake"


def bmw_all_data_stem() -> str:
    return "bmw_OG_remake"


def _threshold_map(prefix: str) -> dict[str, tuple[pd.Series, str]]:
    """Threshold curves for Tesla (prefix='') or BMW (prefix='bmw')."""
    if prefix:
        all_stem = bmw_all_data_stem()
        stems = [
            (all_stem, "All data", COLOR_ALL_DATA),
            (f"{prefix}_OG_remake_90", "All data threshold 0.9", COLOR_90),
            (f"{prefix}_OG_remake_95", "All data threshold 0.95", COLOR_95),
            (f"{prefix}_OG_remake_98", "All data threshold 0.98", COLOR_98),
        ]
    else:
        stems = [
            (tesla_all_data_stem(), "All data", COLOR_ALL_DATA),
            ("OG_remake_90", "All data threshold 0.9", COLOR_90),
            ("OG_remake_95", "All data threshold 0.95", COLOR_95),
            ("OG_remake_98", "All data threshold 0.98", COLOR_98),
        ]
    return build_series_map(stems)


def _allando_map(prefix: str = "") -> dict[str, tuple[pd.Series, str]]:
    p = f"{prefix}_" if prefix else ""
    all_stem = bmw_all_data_stem() if prefix else tesla_all_data_stem()
    return build_series_map(
        [
            (all_stem, "All data", COLOR_ALL_DATA),
            (f"{p}allando_chirp", "Chirp (constant speed)", COLOR_CHIRP),
            (f"{p}allando_savvaltas", "Lane change (constant speed)", COLOR_LANE_CS),
            (f"{p}allando_sin", "Sinusoidal (constant speed)", COLOR_SIN_CS),
        ]
    )


def _valtozo_map(prefix: str = "") -> dict[str, tuple[pd.Series, str]]:
    p = f"{prefix}_" if prefix else ""
    all_stem = bmw_all_data_stem() if prefix else tesla_all_data_stem()
    return build_series_map(
        [
            (all_stem, "All data", COLOR_ALL_DATA),
            (f"{p}valtozo_savvaltas_gas", "Lane change (acceleration)", COLOR_LANE_ACC),
            (f"{p}valtozo_savvaltas_fek", "Lane change (slow down)", COLOR_LANE_SLOW),
            (f"{p}valtozo_sin_gas", "Sinusoidal (acceleration)", COLOR_SIN_ACC),
            (f"{p}valtozo_sin_fek", "Sinusoidal (slow down)", COLOR_SIN_SLOW),
        ]
    )


def main() -> None:
    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tesla_thresholds = _threshold_map("")
    bmw_thresholds = _threshold_map("bmw")
    tesla_allando = _allando_map("")
    bmw_allando = _allando_map("bmw")
    tesla_valtozo = _valtozo_map("")
    bmw_valtozo = _valtozo_map("bmw")

    # 6 single-panel figures
    plot_single(tesla_thresholds, OUT_DIR / "val_thresholds_tesla.png")
    plot_single(bmw_thresholds, OUT_DIR / "val_thresholds_bmw.png")
    plot_single(tesla_allando, OUT_DIR / "val_allando_tesla.png")
    plot_single(bmw_allando, OUT_DIR / "val_allando_bmw.png")
    plot_single(tesla_valtozo, OUT_DIR / "val_valtozo_tesla.png")
    plot_single(bmw_valtozo, OUT_DIR / "val_valtozo_bmw.png")

    # 3 two-panel figures
    plot_two_panel(tesla_thresholds, bmw_thresholds, OUT_DIR / "val_thresholds_combined.png")
    plot_two_panel(tesla_allando, tesla_valtozo, OUT_DIR / "val_speed_tesla.png")
    plot_two_panel(bmw_allando, bmw_valtozo, OUT_DIR / "val_speed_bmw.png")

    print(f"\nDone. 9 figures (PNG + EPS) written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
