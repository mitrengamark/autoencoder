"""
Export MATLAB .fig files to PNG and EPS using matplotlib.

Reads figure data via scipy.io.loadmat and replots lines, scatter, labels,
grid, limits, and legend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

MAT_DIR = Path("mat/figs")
DPI = 300

# Match plot_val_losses_readable.py (Jurnal_plots/readable/)
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 20

TRACK_BLUE = (0.0, 0.0, 1.0)

MATLAB_LINE_STYLE = {
    "-": "-",
    "--": "--",
    "-.": "-.",
    ":": ":",
    "none": "None",
}


def _as_array(value):
    if value is None:
        return None
    arr = np.atleast_1d(value)
    if arr.dtype == object:
        return arr
    return arr


def _mat_color(color) -> tuple[float, float, float] | None:
    if color is None:
        return None
    arr = np.asarray(color, dtype=float).ravel()
    if arr.size >= 3:
        return tuple(arr[:3])
    return None


def _mat_strings(value) -> list[str]:
    if value is None:
        return []
    arr = _as_array(value)
    return [str(v) for v in arr.tolist()]


def _format_axis_label(text: str) -> str:
    """Convert MATLAB-style labels to matplotlib mathtext."""
    if not text:
        return text
    replacements = {
        "m/s^2": r"m/s$^2$",
        "m/s^ 2": r"m/s$^2$",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _line_style(style: str | None) -> str:
    if style is None:
        return "-"
    return MATLAB_LINE_STYLE.get(str(style), "-")


def _collect_axis_labels(ax_struct, xlim, ylim) -> tuple[str | None, str | None]:
    xlabel = None
    ylabel = None
    x_center = np.mean(xlim)
    y_center = np.mean(ylim)
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]

    for child in _as_array(ax_struct.children):
        if getattr(child, "type", None) != "text":
            continue
        props = child.properties
        text = getattr(props, "String", None)
        if text is None or str(text).strip() in {"", "None"}:
            continue
        pos = np.asarray(getattr(props, "Position", [0, 0, 0]), dtype=float).ravel()
        if pos.size < 2:
            continue
        x_pos, y_pos = pos[0], pos[1]

        if abs(x_pos - x_center) <= 0.6 * x_span and y_pos < ylim[0] + 0.08 * y_span:
            xlabel = str(text)
        elif abs(y_pos - y_center) <= 0.6 * y_span and x_pos < xlim[0] + 0.08 * x_span:
            ylabel = str(text)

    return xlabel, ylabel


def _extract_plot_elements(ax_struct):
    lines: list[dict] = []
    scatters: list[dict] = []

    for child in _as_array(ax_struct.children):
        child_type = getattr(child, "type", None)
        props = child.properties

        if child_type == "graph2d.lineseries":
            x = np.asarray(getattr(props, "XData", []), dtype=float).ravel()
            y = np.asarray(getattr(props, "YData", []), dtype=float).ravel()
            if x.size == 0 or y.size == 0:
                continue
            lines.append(
                {
                    "x": x,
                    "y": y,
                    "label": str(getattr(props, "DisplayName", "") or ""),
                    "color": _mat_color(getattr(props, "Color", None)),
                    "linewidth": float(getattr(props, "LineWidth", 1.0) or 1.0),
                    "linestyle": _line_style(getattr(props, "LineStyle", "-")),
                }
            )
        elif child_type == "specgraph.scattergroup":
            x = np.asarray(getattr(props, "XData", []), dtype=float).ravel()
            y = np.asarray(getattr(props, "YData", []), dtype=float).ravel()
            if x.size == 0 or y.size == 0:
                continue
            marker = getattr(props, "Marker", "o")
            if str(marker).lower() in {"none", ""}:
                marker = "o"
            scatters.append(
                {
                    "x": x,
                    "y": y,
                    "label": str(getattr(props, "DisplayName", "") or ""),
                    "color": _mat_color(getattr(props, "MarkerEdgeColor", None))
                    or _mat_color(getattr(props, "MarkerFaceColor", None))
                    or _mat_color(getattr(props, "CData", None))
                    or TRACK_BLUE,
                    "marker": str(marker),
                    "size": float(getattr(props, "SizeData", 20) or 20),
                }
            )

    return lines, scatters


def _extract_legend(fig_struct):
    for child in _as_array(fig_struct.children):
        if getattr(child, "type", None) != "scribe.legend":
            continue
        props = child.properties
        return {
            "labels": _mat_strings(getattr(props, "String", None)),
            "fontsize": float(getattr(props, "FontSize", 10) or 10),
        }
    return None


def load_matlab_figure(fig_path: Path) -> dict:
    data = sio.loadmat(fig_path, squeeze_me=True, struct_as_record=False)
    if "hgS_070000" not in data:
        raise ValueError(f"No figure structure found in {fig_path}")

    fig_struct = data["hgS_070000"]
    axes_list = []
    for child in _as_array(fig_struct.children):
        if getattr(child, "type", None) != "axes":
            continue
        props = child.properties
        xlim = np.asarray(getattr(props, "XLim", [0, 1]), dtype=float).ravel()
        ylim = np.asarray(getattr(props, "YLim", [0, 1]), dtype=float).ravel()
        lines, scatters = _extract_plot_elements(child)
        xlabel, ylabel = _collect_axis_labels(child, xlim, ylim)
        axes_list.append(
            {
                "xlim": tuple(xlim),
                "ylim": tuple(ylim),
                "xlabel": xlabel,
                "ylabel": ylabel,
                "grid": bool(getattr(props, "XGrid", "off") == "on"),
                "box": bool(getattr(props, "Box", "off") == "on"),
                "fontsize": float(getattr(props, "FontSize", 12) or 12),
                "lines": lines,
                "scatters": scatters,
            }
        )

    if not axes_list:
        raise ValueError(f"No axes found in {fig_path}")

    return {
        "axes": axes_list,
        "legend": _extract_legend(fig_struct),
    }


def _plot_elements(ax, axis_data: dict) -> list:
    handles = []
    labels = []

    for scatter in axis_data["scatters"]:
        size = max(scatter["size"] * 0.5, 4.0)
        handle = ax.scatter(
            scatter["x"],
            scatter["y"],
            s=size,
            c=[scatter["color"]] if scatter["color"] else "C0",
            marker=scatter["marker"],
            linewidths=0.8,
            rasterized=True,
            label=scatter["label"] or None,
        )
        if scatter["label"]:
            handles.append(handle)
            labels.append(scatter["label"])

    for line in axis_data["lines"]:
        (handle,) = ax.plot(
            line["x"],
            line["y"],
            color=line["color"],
            linewidth=line["linewidth"],
            linestyle=line["linestyle"],
            label=line["label"] or None,
        )
        if line["label"]:
            handles.append(handle)
            labels.append(line["label"])

    ax.set_xlim(axis_data["xlim"])
    ax.set_ylim(axis_data["ylim"])
    if axis_data["xlabel"]:
        ax.set_xlabel(_format_axis_label(axis_data["xlabel"]), fontsize=AXIS_LABEL_FONTSIZE)
    if axis_data["ylabel"]:
        ax.set_ylabel(_format_axis_label(axis_data["ylabel"]), fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(axis_data["grid"])
    if axis_data["box"]:
        ax.set_frame_on(True)

    return handles, labels


def export_fig(fig_path: Path, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or fig_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_data = load_matlab_figure(fig_path)

    fig, ax = plt.subplots(figsize=(10, 7))
    axis_data = figure_data["axes"][0]
    handles, labels = _plot_elements(ax, axis_data)

    legend_data = figure_data["legend"]
    if legend_data and legend_data["labels"]:
        legend_labels = legend_data["labels"]
        if labels:
            label_to_handle = dict(zip(labels, handles))
            ordered_handles = []
            ordered_labels = []
            for label in legend_labels:
                if label in label_to_handle:
                    ordered_handles.append(label_to_handle[label])
                    ordered_labels.append(label)
            if ordered_handles:
                ax.legend(
                    ordered_handles,
                    ordered_labels,
                    prop={"size": LEGEND_FONTSIZE},
                )
            else:
                ax.legend(prop={"size": LEGEND_FONTSIZE})
        else:
            ax.legend(prop={"size": LEGEND_FONTSIZE})
    elif labels:
        ax.legend(prop={"size": LEGEND_FONTSIZE})

    fig.tight_layout()

    stem = fig_path.stem
    png_path = out_dir / f"{stem}.png"
    eps_path = out_dir / f"{stem}.eps"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.close(fig)
    return png_path, eps_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MATLAB .fig files to PNG and EPS.")
    parser.add_argument(
        "fig_files",
        nargs="*",
        default=[
            "error_yawrate.fig",
            "Force_slip_2.fig",
            "Froce_slip.fig",
            "lat_acc.fig",
            "track.fig",
            "vy_Error.fig",
        ],
        help="FIG file names or paths (default: all standard mat/*.fig files)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MAT_DIR,
        help="Output directory (default: mat/)",
    )
    args = parser.parse_args()

    for fig_file in args.fig_files:
        fig_path = Path(fig_file)
        if not fig_path.is_file():
            fig_path = MAT_DIR / fig_file
        if not fig_path.is_file():
            raise FileNotFoundError(f"Missing FIG file: {fig_file}")

        png_path, eps_path = export_fig(fig_path, args.out_dir)
        print(f"Saved: {png_path}")
        print(f"Saved: {eps_path}")

    print(f"\nDone. Figures written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
