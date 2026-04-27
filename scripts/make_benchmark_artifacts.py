"""Build paper-ready tables and figures from benchmark.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", type=Path, required=True, help="JSON from scripts/benchmark.py")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def _to_dataframe(report: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for model in report.get("models", []):
        agg = model.get("aggregate", {})
        rows.append(
            {
                "model": model.get("name"),
                "family": model.get("family"),
                "latency_mean_ms": agg.get("latency_mean_ms", {}).get("mean"),
                "latency_mean_ms_ci95": agg.get("latency_mean_ms", {}).get("ci95"),
                "fps_mean": agg.get("fps_mean", {}).get("mean"),
                "fps_mean_ci95": agg.get("fps_mean", {}).get("ci95"),
                "map_50_95": agg.get("mAP_50_95", {}).get("mean"),
                "map_50_95_ci95": agg.get("mAP_50_95", {}).get("ci95"),
                "map_50": agg.get("mAP_50", {}).get("mean"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["map_50_95", "fps_mean"], ascending=[False, False], na_position="last")
    return df


def _save_latex(df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "model",
        "family",
        "latency_mean_ms",
        "latency_mean_ms_ci95",
        "fps_mean",
        "fps_mean_ci95",
        "map_50_95",
        "map_50_95_ci95",
        "map_50",
    ]
    out_path.write_text(
        df[cols].to_latex(
            index=False,
            float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "",
            caption="Real-time pothole instance-segmentation benchmark (6 models, mean ± 95% CI).",
            label="tab:benchmark6",
        )
    )


def _plot_latency_fps(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()
    x = range(len(df))

    ax1.bar([i - 0.2 for i in x], df["latency_mean_ms"], width=0.4, label="Latency mean (ms)")
    ax2.bar([i + 0.2 for i in x], df["fps_mean"], width=0.4, color="orange", label="FPS mean")

    ax1.errorbar(
        [i - 0.2 for i in x],
        df["latency_mean_ms"],
        yerr=df["latency_mean_ms_ci95"],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax2.errorbar(
        [i + 0.2 for i in x],
        df["fps_mean"],
        yerr=df["fps_mean_ci95"],
        fmt="none",
        ecolor="black",
        capsize=3,
    )

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["model"], rotation=25, ha="right")
    ax1.set_ylabel("Latency (ms)")
    ax2.set_ylabel("FPS")
    ax1.set_title("Latency vs FPS (mean ± 95% CI)")
    ax1.grid(axis="y", alpha=0.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_accuracy_speed(df: pd.DataFrame, out_path: Path) -> None:
    acc_df = df[df["map_50_95"].notna()].copy()
    if acc_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in acc_df.iterrows():
        ax.errorbar(
            row["fps_mean"],
            row["map_50_95"],
            xerr=row.get("fps_mean_ci95", 0.0),
            yerr=row.get("map_50_95_ci95", 0.0),
            fmt="o",
            capsize=3,
        )
        ax.annotate(row["model"], (row["fps_mean"], row["map_50_95"]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("FPS mean")
    ax.set_ylabel("COCO mAP@[.5:.95] (segm)")
    ax.set_title("Speed/accuracy frontier (mean ± 95% CI)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report = json.loads(args.benchmark.read_text())
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _to_dataframe(report)
    if df.empty:
        raise SystemExit("No models found in benchmark JSON.")

    csv_path = out_dir / "benchmark_table.csv"
    tex_path = out_dir / "benchmark_table.tex"
    plot_latency_path = out_dir / "benchmark_latency_fps.png"
    plot_frontier_path = out_dir / "benchmark_speed_accuracy.png"

    df.to_csv(csv_path, index=False)
    _save_latex(df, tex_path)
    _plot_latency_fps(df, plot_latency_path)
    _plot_accuracy_speed(df, plot_frontier_path)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {tex_path}")
    print(f"Wrote: {plot_latency_path}")
    if plot_frontier_path.exists():
        print(f"Wrote: {plot_frontier_path}")


if __name__ == "__main__":
    main()
