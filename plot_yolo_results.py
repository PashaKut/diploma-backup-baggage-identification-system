from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


PLOT_COLUMNS = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Ultralytics-style training metrics figure from a results.csv file."
    )
    parser.add_argument("results_csv", type=Path, help="Path to YOLO results.csv")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to results.png next to the CSV.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a preview window after saving the figure.",
    )
    return parser.parse_args()


def read_results(results_csv: Path) -> tuple[list[float], dict[str, list[float]]]:
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    epochs: list[float] = []
    series: dict[str, list[float]] = {column: [] for column in PLOT_COLUMNS}

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [column for column in PLOT_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(
                "results.csv is missing expected Ultralytics columns: " + ", ".join(missing_columns)
            )

        if "epoch" not in (reader.fieldnames or []):
            raise ValueError("results.csv is missing the epoch column.")

        for row in reader:
            epochs.append(float(row["epoch"]))
            for column in PLOT_COLUMNS:
                value = row.get(column, "")
                series[column].append(float(value) if value not in {"", None} else float("nan"))

    if not epochs:
        raise ValueError(f"No data rows found in: {results_csv}")

    return epochs, series


def create_figure(epochs: list[float], series: dict[str, list[float]]):
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    figure, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    axes_flat = axes.flatten()

    for axis, column in zip(axes_flat, PLOT_COLUMNS):
        values = series[column]
        axis.plot(epochs, values, color="#1f77b4", linewidth=1.8, marker=".", markersize=5)
        axis.set_title(column, fontsize=10)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.25)
        axis.tick_params(labelsize=8)

    for axis in axes_flat[len(PLOT_COLUMNS):]:
        axis.axis("off")

    return figure


def try_ultralytics_plot(results_csv: Path, output_path: Path) -> bool:
    try:
        from ultralytics.utils.plotting import plot_results
    except Exception:
        return False

    try:
        plot_results(file=str(results_csv), dir=str(results_csv.parent))
    except TypeError:
        try:
            plot_results(str(results_csv))
        except Exception:
            return False
    except Exception:
        return False

    default_output = results_csv.with_name("results.png")
    if default_output.exists() and output_path != default_output:
        shutil.copy2(default_output, output_path)
    return output_path.exists()


def main() -> None:
    args = parse_args()
    results_csv = args.results_csv.resolve()
    output_path = args.output.resolve() if args.output is not None else results_csv.with_name("results.png")

    if not try_ultralytics_plot(results_csv, output_path):
        epochs, series = read_results(results_csv)

        figure = create_figure(epochs, series)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")

    if args.show:
        import matplotlib.pyplot as plt

        preview = plt.imread(output_path)
        plt.figure(figsize=(20, 8))
        plt.imshow(preview)
        plt.axis("off")
        plt.show()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
