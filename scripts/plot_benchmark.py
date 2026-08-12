#!/usr/bin/env python3
"""Two figures from results/benchmark.json: sequential-write latency
distribution, and throughput vs. concurrent client count -- the second
one is the interesting one (see README, "Benchmarks": throughput peaks
around 2 concurrent clients and then DEGRADES as concurrency increases,
which is a real, reproducible, architecturally-explained result, not
noise).

    python3 scripts/plot_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "benchmark.json"
FIGURES_DIR = ROOT / "figures"

COLOR_PRIMARY = "#0072B2"
COLOR_ACCENT = "#D55E00"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def main() -> None:
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    seq = data["sequential"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.hist(seq["raw_ms"], bins=30, color=COLOR_PRIMARY, edgecolor="white", linewidth=0.4)
    ax.axvline(seq["p95_ms"], color=COLOR_ACCENT, linewidth=1.5, linestyle="--", label=f"p95 = {seq['p95_ms']:.1f} ms")
    ax.set_xlabel("Latency per sequential PUT (ms) -- one full consensus round trip")
    ax.set_ylabel(f"Requests (n={seq['n_ops']})")
    ax.set_title("Single-write latency: JSON-over-TCP, one leader, no batching")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.savefig(FIGURES_DIR / "benchmark_latency.svg")
    plt.close(fig)

    sweep = data["concurrency_sweep"]
    n_clients = [s["n_clients"] for s in sweep]
    ops_per_sec = [s["ops_per_second"] for s in sweep]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(n_clients, ops_per_sec, color=COLOR_PRIMARY, marker="o", markersize=6, linewidth=2.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_clients)
    ax.set_xticklabels([str(n) for n in n_clients])
    ax.set_xlabel("Concurrent clients")
    ax.set_ylabel("Throughput (ops/s)")
    ax.set_title("Throughput peaks around 2 clients, then degrades -- see README")
    fig.savefig(FIGURES_DIR / "benchmark_throughput.svg")
    plt.close(fig)

    print(f"Wrote {FIGURES_DIR / 'benchmark_latency.svg'} and {FIGURES_DIR / 'benchmark_throughput.svg'}")


if __name__ == "__main__":
    main()
