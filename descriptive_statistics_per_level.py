import sys
import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from descriptive_statistics import (
    load_phone_data, load_lab_data, clean_phone_data,
    SINGLE_IDS, MULTIPLE_IDS, OUTPUT_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})

LEVELS = [1, 2, 3]
LEVELS_10 = list(range(1, 11))

PER_LEVEL_PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "per_level_plots")


def print_separator(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def subsection(title):
    print(f"\n  --- {title} ---\n")


def build_per_level_rt(phone_single, phone_multiple, lab_single, lab_multiple):
    frames = []

    for phone_df in [phone_single, phone_multiple]:
        subset = phone_df[
            (phone_df["Completed"] == True)
            & (phone_df["is_retry"] == False)
            & (phone_df["Level"].isin(LEVELS))
        ].copy()
        subset = subset.rename(columns={
            "InitialResponseTime(ms)": "rt_ms",
            "Level": "level",
        })
        frames.append(subset[["participant_id", "target_count", "device", "level", "rt_ms"]])

    for lab_df in [lab_single, lab_multiple]:
        subset = lab_df[lab_df["trials.thisN"].isin([0, 1, 2])].copy()
        subset["level"] = subset["trials.thisN"].astype(int) + 1
        subset = subset.rename(columns={"response_time_ms": "rt_ms"})
        frames.append(subset[["participant_id", "target_count", "device", "level", "rt_ms"]])

    return pd.concat(frames, ignore_index=True)


def build_per_level_accuracy(phone_single, phone_multiple):
    frames = []
    for phone_df in [phone_single, phone_multiple]:
        subset = phone_df[
            (phone_df["Completed"] == True)
            & (phone_df["is_retry"] == False)
            & (phone_df["Level"].isin(LEVELS))
        ].copy()
        subset = subset.rename(columns={
            "Level": "level",
            "SuccessRate(%)": "success_rate",
            "HitRate(%)": "hit_rate",
            "FalseAlarms": "false_alarms",
        })
        frames.append(subset[["participant_id", "target_count", "level",
                               "success_rate", "hit_rate", "false_alarms"]])
    return pd.concat(frames, ignore_index=True)


def _phone_participants_with_level10(phone_df):
    completed_any = phone_df[phone_df["Completed"] == True]
    return completed_any[completed_any["Level"] >= 10]["participant_id"].unique()


def _first_completed_per_level(phone_df, pids, levels):
    completed_any = phone_df[phone_df["Completed"] == True]
    subset = completed_any[
        completed_any["participant_id"].isin(pids)
        & completed_any["Level"].isin(levels)
    ].copy()
    subset = subset.sort_values(["participant_id", "Level", "Timestamp"])
    subset = subset.drop_duplicates(subset=["participant_id", "Level"], keep="first")
    return subset


def build_10level_rt(phone_single, phone_multiple, lab_single, lab_multiple):
    frames = []

    for phone_df in [phone_single, phone_multiple]:
        pids_10 = _phone_participants_with_level10(phone_df)
        subset = _first_completed_per_level(phone_df, pids_10, LEVELS_10)
        subset = subset.rename(columns={
            "InitialResponseTime(ms)": "rt_ms",
            "Level": "level",
        })
        frames.append(subset[["participant_id", "target_count", "device", "level", "rt_ms"]])

    for lab_df in [lab_single, lab_multiple]:
        subset = lab_df[lab_df["trials.thisN"].isin(range(10))].copy()
        subset["level"] = subset["trials.thisN"].astype(int) + 1
        subset = subset.rename(columns={"response_time_ms": "rt_ms"})
        frames.append(subset[["participant_id", "target_count", "device", "level", "rt_ms"]])

    return pd.concat(frames, ignore_index=True)


def build_phone_10level_accuracy(phone_single, phone_multiple):
    frames = []
    for phone_df in [phone_single, phone_multiple]:
        pids_10 = _phone_participants_with_level10(phone_df)
        subset = _first_completed_per_level(phone_df, pids_10, LEVELS_10)
        subset = subset.rename(columns={
            "Level": "level",
            "SuccessRate(%)": "success_rate",
            "HitRate(%)": "hit_rate",
            "FalseAlarms": "false_alarms",
        })
        frames.append(subset[["participant_id", "target_count", "level",
                               "success_rate", "hit_rate", "false_alarms"]])
    return pd.concat(frames, ignore_index=True)


def print_per_level_rt_tables(per_level):
    print_separator("Table 1: Per-Level Response Time (ms) Descriptive Statistics")

    for lvl in LEVELS:
        subsection(f"Level {lvl}")
        subset = per_level[per_level["level"] == lvl]
        stats = subset.groupby(["target_count", "device"])["rt_ms"].agg(
            N="count", Mean="mean", Median="median", SD="std", Min="min", Max="max"
        ).round(1)
        print(stats.to_string())


def print_cross_level_trend(per_level):
    print_separator("Table 2: Cross-Level RT Trend (Mean RT across Levels 1-3)")

    rows = []
    for tc in ["single", "multiple"]:
        for dev in ["phone", "lab"]:
            means = []
            for lvl in LEVELS:
                subset = per_level[
                    (per_level["target_count"] == tc)
                    & (per_level["device"] == dev)
                    & (per_level["level"] == lvl)
                ]
                means.append(subset["rt_ms"].mean())

            if means[0] > 0:
                pct_change = ((means[2] - means[0]) / means[0]) * 100
            else:
                pct_change = 0.0
            arrow = "UP" if pct_change > 5 else ("DOWN" if pct_change < -5 else "FLAT")

            rows.append({
                "Condition": f"{tc}/{dev}",
                "Level 1": f"{means[0]:.1f}",
                "Level 2": f"{means[1]:.1f}",
                "Level 3": f"{means[2]:.1f}",
                "Direction": arrow,
                "% Change (1->3)": f"{pct_change:+.1f}%",
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def print_per_level_accuracy(acc_df):
    print_separator("Table 3: Per-Level Accuracy Metrics (Phone Only)")

    for lvl in LEVELS:
        subsection(f"Level {lvl}")
        subset = acc_df[acc_df["level"] == lvl]
        stats = subset.groupby("target_count").agg(
            N=("participant_id", "count"),
            Mean_SuccessRate=("success_rate", "mean"),
            SD_SuccessRate=("success_rate", "std"),
            Mean_HitRate=("hit_rate", "mean"),
            SD_HitRate=("hit_rate", "std"),
            Mean_FalseAlarms=("false_alarms", "mean"),
            SD_FalseAlarms=("false_alarms", "std"),
        ).round(2)
        print(stats.to_string())


def print_per_level_correlation(per_level):
    print_separator("Table 4: Per-Level Phone-Lab RT Correlation (RQ1)")

    rows = []
    for tc in ["single", "multiple"]:
        for lvl in LEVELS:
            phone = per_level[
                (per_level["target_count"] == tc)
                & (per_level["device"] == "phone")
                & (per_level["level"] == lvl)
            ].set_index("participant_id")["rt_ms"]

            lab = per_level[
                (per_level["target_count"] == tc)
                & (per_level["device"] == "lab")
                & (per_level["level"] == lvl)
            ].set_index("participant_id")["rt_ms"]

            common = phone.index.intersection(lab.index)
            n = len(common)
            if n >= 3:
                r, p = pearsonr(phone.loc[common], lab.loc[common])
                p_str = f"{p:.4f}" if p >= 0.001 else "<.001"
                rows.append({
                    "Condition": f"{tc}",
                    "Level": lvl,
                    "n": n,
                    "r": f"{r:.3f}",
                    "p": p_str,
                })
            else:
                rows.append({
                    "Condition": f"{tc}",
                    "Level": lvl,
                    "n": n,
                    "r": "N/A",
                    "p": "N/A",
                })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def plot_per_level_rt_boxplots(per_level):
    for lvl in LEVELS:
        fig, ax = plt.subplots(figsize=(7, 5))
        subset = per_level[per_level["level"] == lvl]

        sns.boxplot(data=subset, x="target_count", y="rt_ms", hue="device",
                    palette="Set2", ax=ax)
        sns.stripplot(data=subset, x="target_count", y="rt_ms", hue="device",
                      palette="Set2", dodge=True, alpha=0.5, size=5, ax=ax,
                      legend=False)

        ax.set_title(f"Level {lvl} — Response Time Distribution by Condition")
        ax.set_xlabel("Target Count")
        ax.set_ylabel("Response Time (ms)")
        ax.legend(title="Device", fontsize=9)

        plt.tight_layout()
        fname = f"per_level_rt_boxplot_level{lvl}.png"
        fig.savefig(os.path.join(PER_LEVEL_PLOT_DIR, fname))
        plt.close(fig)
        print(f"    Saved: {fname}")


def plot_per_level_paired_lines(per_level):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey="row")

    for row_idx, tc in enumerate(["single", "multiple"]):
        for col_idx, lvl in enumerate(LEVELS):
            ax = axes[row_idx, col_idx]
            subset = per_level[
                (per_level["target_count"] == tc)
                & (per_level["level"] == lvl)
            ]
            pivot = subset.pivot(index="participant_id", columns="device", values="rt_ms")
            if "phone" not in pivot.columns or "lab" not in pivot.columns:
                ax.set_title(f"{tc} / Level {lvl}\n(no paired data)")
                continue
            pivot = pivot.dropna()

            for _, r in pivot.iterrows():
                ax.plot(["phone", "lab"], [r["phone"], r["lab"]],
                        color="gray", alpha=0.4, linewidth=1)
            ax.scatter(["phone"] * len(pivot), pivot["phone"],
                       color="steelblue", s=25, zorder=3)
            ax.scatter(["lab"] * len(pivot), pivot["lab"],
                       color="coral", s=25, zorder=3)
            ax.set_title(f"{tc.capitalize()} / Level {lvl} (n={len(pivot)})")
            ax.set_xlabel("Device")
            if col_idx == 0:
                ax.set_ylabel("Response Time (ms)")

    plt.suptitle("Per-Level Within-Subject Device Comparison", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(PER_LEVEL_PLOT_DIR, "per_level_paired_lines.png"))
    plt.close(fig)


def plot_rt_trends(rt_10level):
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {
        ("single",   "phone"): "#2196F3",
        ("single",   "lab"):   "#64B5F6",
        ("multiple", "phone"): "#FF5722",
        ("multiple", "lab"):   "#FF8A65",
    }
    markers = {
        ("single",   "phone"): "o",
        ("single",   "lab"):   "s",
        ("multiple", "phone"): "^",
        ("multiple", "lab"):   "D",
    }
    linestyles = {
        ("single",   "phone"): "-",
        ("single",   "lab"):   "--",
        ("multiple", "phone"): "-",
        ("multiple", "lab"):   "--",
    }

    for tc in ["single", "multiple"]:
        for dev in ["phone", "lab"]:
            subset = rt_10level[
                (rt_10level["target_count"] == tc) & (rt_10level["device"] == dev)
            ]
            n_participants = subset["participant_id"].nunique()
            means, sems = [], []
            for lvl in LEVELS_10:
                vals = subset[subset["level"] == lvl]["rt_ms"]
                means.append(vals.mean() if len(vals) > 0 else np.nan)
                sems.append(vals.sem() if len(vals) > 0 else np.nan)

            key = (tc, dev)
            ax.errorbar(LEVELS_10, means, yerr=sems,
                        label=f"{tc}/{dev} (n={n_participants})",
                        color=colors[key], marker=markers[key],
                        linestyle=linestyles[key],
                        markersize=6, linewidth=2, capsize=3, capthick=1.2)

    ax.set_xlabel("Level")
    ax.set_ylabel("Mean Response Time (ms)")
    ax.set_title(
        "RT Trend Across Levels 1-10 — Phone & Lab\n"
        "(Phone: participants with all 10 levels; Lab: all participants; error bars = SEM)"
    )
    ax.set_xticks(LEVELS_10)
    ax.legend(title="Condition", fontsize=9, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(PER_LEVEL_PLOT_DIR, "per_level_rt_trends.png"))
    plt.close(fig)


def plot_accuracy_trends(acc_10level):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"single": "steelblue", "multiple": "coral"}

    for tc in ["single", "multiple"]:
        subset = acc_10level[acc_10level["target_count"] == tc]
        n_participants = subset["participant_id"].nunique()

        means_s, sems_s = [], []
        means_h, sems_h = [], []
        for lvl in LEVELS_10:
            vals_s = subset[subset["level"] == lvl]["success_rate"]
            vals_h = subset[subset["level"] == lvl]["hit_rate"]
            means_s.append(vals_s.mean() if len(vals_s) > 0 else np.nan)
            sems_s.append(vals_s.sem() if len(vals_s) > 0 else np.nan)
            means_h.append(vals_h.mean() if len(vals_h) > 0 else np.nan)
            sems_h.append(vals_h.sem() if len(vals_h) > 0 else np.nan)

        axes[0].errorbar(LEVELS_10, means_s, yerr=sems_s,
                         label=f"{tc} (n={n_participants})",
                         color=colors[tc], marker="o", markersize=7,
                         linewidth=2, capsize=4, capthick=1.5)
        axes[1].errorbar(LEVELS_10, means_h, yerr=sems_h,
                         label=f"{tc} (n={n_participants})",
                         color=colors[tc], marker="s", markersize=7,
                         linewidth=2, capsize=4, capthick=1.5)

    for ax, metric in zip(axes, ["Success Rate (%)", "Hit Rate (%)"]):
        ax.set_xlabel("Level")
        ax.set_ylabel(f"Mean {metric}")
        ax.set_title(f"{metric} by Level (Phone)")
        ax.set_xticks(LEVELS_10)
        ax.legend(title="Target Count")

    plt.suptitle(
        "Per-Level Accuracy Trends Across Levels 1-10 — Phone Only\n"
        "(Lab has no accuracy metrics; phone: participants with all 10 levels)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(PER_LEVEL_PLOT_DIR, "per_level_accuracy_trends.png"))
    plt.close(fig)


def plot_per_level_correlation(per_level):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row_idx, tc in enumerate(["single", "multiple"]):
        for col_idx, lvl in enumerate(LEVELS):
            ax = axes[row_idx, col_idx]

            phone = per_level[
                (per_level["target_count"] == tc)
                & (per_level["device"] == "phone")
                & (per_level["level"] == lvl)
            ].set_index("participant_id")["rt_ms"]

            lab = per_level[
                (per_level["target_count"] == tc)
                & (per_level["device"] == "lab")
                & (per_level["level"] == lvl)
            ].set_index("participant_id")["rt_ms"]

            common = phone.index.intersection(lab.index)
            if len(common) < 3:
                ax.set_title(f"{tc} / Level {lvl}\n(n={len(common)}, too few)")
                ax.set_xlabel("Phone RT (ms)")
                ax.set_ylabel("Lab RT (ms)")
                continue

            phone_vals = phone.loc[common].values
            lab_vals = lab.loc[common].values

            ax.scatter(phone_vals, lab_vals, color="steelblue", s=35,
                       edgecolors="white", zorder=3)

            r, p = pearsonr(phone_vals, lab_vals)
            slope, intercept = np.polyfit(phone_vals, lab_vals, 1)
            x_line = np.linspace(phone_vals.min(), phone_vals.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, "-", color="coral",
                    linewidth=2)

            all_vals = np.concatenate([phone_vals, lab_vals])
            lims = [all_vals.min() * 0.9, all_vals.max() * 1.1]
            ax.plot(lims, lims, "--", color="gray", alpha=0.4)

            p_str = f"p={p:.3f}" if p >= 0.001 else "p<.001"
            ax.annotate(f"r={r:.2f}, {p_str}\nn={len(common)}",
                        xy=(0.05, 0.95), xycoords="axes fraction",
                        fontsize=9, ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

            ax.set_title(f"{tc.capitalize()} / Level {lvl}")
            ax.set_xlabel("Phone RT (ms)")
            if col_idx == 0:
                ax.set_ylabel("Lab RT (ms)")

    plt.suptitle("Per-Level Phone-Lab RT Correlation (RQ1)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(PER_LEVEL_PLOT_DIR, "per_level_correlation.png"))
    plt.close(fig)


def main():
    os.makedirs(PER_LEVEL_PLOT_DIR, exist_ok=True)

    print("=" * 65)
    print("  PER-LEVEL DESCRIPTIVE STATISTICS: ATTENTION VALIDATION TASK")
    print("=" * 65)

    print("\nLoading data...")
    phone_single = clean_phone_data(load_phone_data("single", SINGLE_IDS))
    phone_multiple = clean_phone_data(load_phone_data("multiple", MULTIPLE_IDS))
    lab_single = load_lab_data("single", SINGLE_IDS)
    lab_multiple = load_lab_data("multiple", MULTIPLE_IDS)

    print("Building per-level dataframes...")
    per_level = build_per_level_rt(phone_single, phone_multiple, lab_single, lab_multiple)
    acc_df = build_per_level_accuracy(phone_single, phone_multiple)
    rt_10level = build_10level_rt(phone_single, phone_multiple, lab_single, lab_multiple)
    acc_10level = build_phone_10level_accuracy(phone_single, phone_multiple)

    print(f"  Per-level RT (levels 1-3, phone+lab): {len(per_level)} rows")
    print(f"  Per-level accuracy (levels 1-3, phone): {len(acc_df)} rows")
    print(f"  10-level RT (phone: participants with >=10 levels; lab: all participants):")
    for dev in ["phone", "lab"]:
        for tc in ["single", "multiple"]:
            n = rt_10level[(rt_10level["target_count"] == tc) & (rt_10level["device"] == dev)]["participant_id"].nunique()
            print(f"    {tc}/{dev}: {n} participants")

    print_per_level_rt_tables(per_level)
    print_cross_level_trend(per_level)
    print_per_level_accuracy(acc_df)
    print_per_level_correlation(per_level)

    print(f"\n{'=' * 65}")
    print(f"  Generating per-level plots -> {PER_LEVEL_PLOT_DIR}")
    print(f"{'=' * 65}")

    print("  [1/5] per_level_rt_boxplot_level{1,2,3}.png")
    plot_per_level_rt_boxplots(per_level)

    plot_per_level_paired_lines(per_level)
    print("  [2/5] per_level_paired_lines.png")

    plot_rt_trends(rt_10level)
    print("  [3/5] per_level_rt_trends.png  (levels 1-10, phone + lab)")

    plot_accuracy_trends(acc_10level)
    print("  [4/5] per_level_accuracy_trends.png  (levels 1-10, phone only)")

    plot_per_level_correlation(per_level)
    print("  [5/5] per_level_correlation.png")

    print(f"\nAll per-level plots saved to: {PER_LEVEL_PLOT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
