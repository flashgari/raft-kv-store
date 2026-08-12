#!/usr/bin/env python3
"""Two figures from results/chaos_campaign.json: recovery-time
distribution, and recovery time vs. injected message-drop probability
(showing the system degrades gracefully rather than catastrophically as
the network gets worse). Okabe-Ito colorblind-safe palette, matching the
convention used across this portfolio's other projects.

    python3 scripts/plot_chaos_campaign.py
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
RESULTS_PATH = ROOT / "results" / "chaos_campaign.json"
FIGURES_DIR = ROOT / "figures"

COLOR_PRIMARY = "#0072B2"  # blue
COLOR_ACCENT = "#D55E00"  # vermillion
COLOR_REFERENCE = "#999999"

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
    trials = [t for t in data["trials"] if t.get("recovered")]
    summary = data["summary"]

    recovery_ms = [t["recovery_ms"] for t in trials]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.hist(recovery_ms, bins=40, color=COLOR_PRIMARY, edgecolor="white", linewidth=0.4)
    ax.axvline(summary["recovery_ms_p95"], color=COLOR_ACCENT, linewidth=1.5, linestyle="--", label=f"p95 = {summary['recovery_ms_p95']:.0f} ms")
    ax.set_xlabel("Time to re-elect a leader after heal + restart (ms, simulated time)")
    ax.set_ylabel("Trials")
    ax.set_title(f"Leader-recovery time across {summary['trials']} randomized chaos trials")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.savefig(FIGURES_DIR / "chaos_recovery_time.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    drop = [t["drop_probability"] for t in trials]
    ax.scatter(drop, recovery_ms, s=10, alpha=0.35, color=COLOR_PRIMARY, edgecolors="none")
    ax.set_xlabel("Injected per-message drop probability for this trial")
    ax.set_ylabel("Recovery time (ms, simulated time)")
    ax.set_title("Recovery time degrades gracefully as the network gets worse")
    fig.savefig(FIGURES_DIR / "chaos_recovery_vs_loss.svg")
    plt.close(fig)

    print(f"Wrote {FIGURES_DIR / 'chaos_recovery_time.svg'} and {FIGURES_DIR / 'chaos_recovery_vs_loss.svg'}")


if __name__ == "__main__":
    main()
