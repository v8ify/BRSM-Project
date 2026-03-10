import sys
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from descriptive_statistics import (
    load_phone_data, load_lab_data, clean_phone_data,
    SINGLE_IDS, MULTIPLE_IDS,
)
from descriptive_statistics_per_level import build_per_level_rt, PER_LEVEL_PLOT_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})

LEVELS = [1, 2, 3]
ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / 2
N_PERMUTATIONS = 10_000


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def subsection(title):
    print(f"\n  --- {title} ---\n")


def sig_marker(p, alpha=ALPHA):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < alpha:
        return "*"
    return "ns"


def _welch_t(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denom = np.sqrt(va / na + vb / nb)
    if denom == 0:
        return 0.0
    return (a.mean() - b.mean()) / denom


def run_permutation_tests_per_level(per_level):
    section("SECTION 1: PERMUTATION TESTS PER LEVEL (10,000 permutations, seed=42)")

    rng = np.random.default_rng(42)
    all_results = []

    for lvl in LEVELS:
        subsection(f"Level {lvl}")
        lv = per_level[per_level["level"] == lvl]

        paired_data = {}
        for tc in ["single", "multiple"]:
            phone_s = lv[(lv["target_count"] == tc) & (lv["device"] == "phone")].set_index("participant_id")["rt_ms"]
            lab_s = lv[(lv["target_count"] == tc) & (lv["device"] == "lab")].set_index("participant_id")["rt_ms"]
            common = phone_s.index.intersection(lab_s.index)
            paired_data[tc] = {
                "phone": phone_s.loc[common].values,
                "lab": lab_s.loc[common].values,
                "diff": phone_s.loc[common].values - lab_s.loc[common].values,
                "n": len(common),
            }

        for tc in ["single", "multiple"]:
            diff = paired_data[tc]["diff"]
            n = paired_data[tc]["n"]
            if n < 2 or diff.std(ddof=1) == 0:
                print(f"  Device effect ({tc}): n={n}, skipped (insufficient data)")
                continue

            observed = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))

            null_dist = np.empty(N_PERMUTATIONS)
            for i in range(N_PERMUTATIONS):
                signs = rng.choice([-1, 1], size=n)
                flipped = signs * diff
                null_dist[i] = flipped.mean() / (flipped.std(ddof=1) / np.sqrt(n))

            p_perm = np.mean(np.abs(null_dist) >= np.abs(observed))
            p_str = "p < 0.0001" if p_perm == 0 else f"p = {p_perm:.4f}"
            eff_p = p_perm if p_perm > 0 else 0.00001

            print(f"  Device effect ({tc}, n={n}):")
            print(f"    Observed t = {observed:.3f}")
            print(f"    Permutation {p_str}  {sig_marker(eff_p, BONFERRONI_ALPHA)}")

            all_results.append({
                "level": lvl,
                "label": f"Device effect ({tc})",
                "observed": observed,
                "p_perm": p_perm,
                "null_distribution": null_dist,
                "test_type": "within",
            })

    return all_results


def print_cross_level_trends(all_results):
    section("SECTION 2: CROSS-LEVEL EFFECT TRENDS (RQ4)")

    labels = ["Device effect (single)", "Device effect (multiple)"]

    rows = []
    for label in labels:
        level_results = [r for r in all_results if r["label"] == label]
        level_results.sort(key=lambda x: x["level"])

        if len(level_results) < 2:
            continue

        t_vals = [f"{r['observed']:.3f}" for r in level_results]
        p_vals = []
        for r in level_results:
            p = r["p_perm"]
            p_vals.append(f"{p:.4f}" if p > 0 else "<.0001")

        obs_list = [r["observed"] for r in level_results]
        if len(obs_list) >= 2:
            abs_change = abs(obs_list[-1]) - abs(obs_list[0])
            if abs_change > 0.5:
                trend = "Growing"
            elif abs_change < -0.5:
                trend = "Shrinking"
            else:
                trend = "Stable"
        else:
            trend = "N/A"

        rows.append({
            "Effect": label,
            "t (L1)": t_vals[0] if len(t_vals) > 0 else "N/A",
            "t (L2)": t_vals[1] if len(t_vals) > 1 else "N/A",
            "t (L3)": t_vals[2] if len(t_vals) > 2 else "N/A",
            "p (L1)": p_vals[0] if len(p_vals) > 0 else "N/A",
            "p (L2)": p_vals[1] if len(p_vals) > 1 else "N/A",
            "p (L3)": p_vals[2] if len(p_vals) > 2 else "N/A",
            "Trend": trend,
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print(df.to_string(index=False))


def plot_permutation_nulls(all_results):
    section("PERMUTATION NULL DISTRIBUTION PLOTS (PER LEVEL)")

    os.makedirs(PER_LEVEL_PLOT_DIR, exist_ok=True)

    test_labels = [
        "Device effect (single)",
        "Device effect (multiple)",
    ]

    for lvl in LEVELS:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(
            f"Level {lvl} — Permutation Null Distributions (2 Tests, {N_PERMUTATIONS:,} permutations)\n"
            "Red dashed = observed t  |  Orange = 2.5th / 97.5th percentile of null",
            fontsize=11,
        )

        perm_results = []
        for label in test_labels:
            matching = [r for r in all_results if r["level"] == lvl and r["label"] == label]
            perm_results.append(matching[0] if matching else None)

        for col_idx, (ax, pr) in enumerate(zip(axes[:2], perm_results)):
            label = test_labels[col_idx]
            if pr is None:
                ax.set_title(f"{label}\n(no data)", fontsize=8)
                ax.axis("off")
                continue

            null = pr["null_distribution"]
            obs = pr["observed"]
            p = pr["p_perm"]

            ax.hist(null, bins=50, color="#4A90D9", alpha=0.7,
                    edgecolor="white", density=True)
            ax.axvline(obs, color="red", linewidth=2, linestyle="--",
                       label=f"Obs = {obs:.2f}")

            tail_pct = BONFERRONI_ALPHA / 2 * 100
            crit_lower = np.percentile(null, tail_pct)
            crit_upper = np.percentile(null, 100 - tail_pct)
            ax.axvline(crit_lower, color="orange", linewidth=1.3, linestyle="-.",
                       label=f"{tail_pct:.2f}%={crit_lower:.2f}")
            ax.axvline(crit_upper, color="orange", linewidth=1.3, linestyle="-.",
                       label=f"{100-tail_pct:.2f}%={crit_upper:.2f}")

            eff_p = p if p > 0 else 0.00001
            sig = eff_p < BONFERRONI_ALPHA
            p_str = "p<.0001" if p == 0 else f"p={p:.4f}"
            box_color = "#ffcccc" if sig else "#e0e0e0"
            text_color = "red" if sig else "gray"
            ax.annotate(
                f"{p_str}\n{'Sig' if sig else 'NS'}",
                xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color=text_color,
                bbox=dict(boxstyle="round,pad=0.3", fc=box_color, alpha=0.9),
            )

            ax.set_title(label, fontsize=9, fontweight="bold")
            ax.set_xlabel("t-value", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="upper left")

        ax_text = axes[2]
        ax_text.axis("off")
        lines = [f"Level {lvl} Results", ""]
        for pr in perm_results:
            if pr is None:
                continue
            p = pr["p_perm"]
            eff_p = p if p > 0 else 0.00001
            sig = eff_p < BONFERRONI_ALPHA
            p_str = "p<.0001" if p == 0 else f"p={p:.4f}"
            marker = sig_marker(eff_p, BONFERRONI_ALPHA)
            lines.append(f"{pr['label']}:")
            lines.append(f"  t={pr['observed']:.3f}, {p_str} {marker}")
            lines.append("")
        lines.append(f"alpha = {BONFERRONI_ALPHA} (Bonferroni)")
        ax_text.text(0.05, 0.95, "\n".join(lines), transform=ax_text.transAxes,
                     fontsize=8.5, va="top", ha="left",
                     bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", alpha=0.9))

        plt.tight_layout()
        fname = f"per_level_permutation_nulls_level{lvl}.png"
        outpath = os.path.join(PER_LEVEL_PLOT_DIR, fname)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def main():
    print("=" * 70)
    print("  PER-LEVEL INFERENTIAL STATISTICS: ATTENTION VALIDATION TASK")
    print("=" * 70)

    os.makedirs(PER_LEVEL_PLOT_DIR, exist_ok=True)

    print("\nLoading data...")
    phone_single = clean_phone_data(load_phone_data("single", SINGLE_IDS))
    phone_multiple = clean_phone_data(load_phone_data("multiple", MULTIPLE_IDS))
    lab_single = load_lab_data("single", SINGLE_IDS)
    lab_multiple = load_lab_data("multiple", MULTIPLE_IDS)

    per_level = build_per_level_rt(phone_single, phone_multiple, lab_single, lab_multiple)
    print(f"  Per-level RT: {len(per_level)} rows")

    all_results = run_permutation_tests_per_level(per_level)

    print_cross_level_trends(all_results)

    plot_permutation_nulls(all_results)

    print(f"\nAll plots saved to: {PER_LEVEL_PLOT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
