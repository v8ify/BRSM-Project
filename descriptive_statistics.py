import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import ast
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_brsm")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

SINGLE_IDS = range(1, 22)
MULTIPLE_IDS = range(22, 38)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})


def load_phone_data(target_count, id_range):
    folder = os.path.join(BASE_DIR, target_count, "phone")
    frames = []
    for pid in id_range:
        filepath = os.path.join(folder, f"{pid}_attentional_spotter_results.csv")
        if not os.path.exists(filepath):
            continue
        df = pd.read_csv(filepath)
        df["participant_id"] = pid
        df["target_count"] = target_count
        df["device"] = "phone"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_lab_data(target_count, id_range):
    folder = os.path.join(BASE_DIR, target_count, "lab")
    frames = []
    for pid in id_range:
        pattern = os.path.join(folder, f"{pid}_visual_search_*.csv")
        matches = glob.glob(pattern)
        if not matches:
            continue
        filepath = matches[0]
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        df = df[df["target_col"].notna() & (df["target_col"].str.strip() != "")].copy()
        df["participant_id"] = pid
        df["target_count"] = target_count
        df["device"] = "lab"

        if target_count == "single":
            df["response_time_ms"] = df["mouse.time"].apply(_parse_rt_single)
        else:
            parsed = df["mouse.time"].apply(_parse_rt_multiple)
            df["response_time_ms"] = parsed.apply(lambda x: x[0])
            df["mean_inter_target_ms"] = parsed.apply(lambda x: x[1])

        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _parse_rt_single(val):
    try:
        parsed = ast.literal_eval(str(val))
        return float(parsed[0]) * 1000
    except (ValueError, SyntaxError, IndexError):
        return np.nan


def _parse_rt_multiple(val):
    try:
        parsed = ast.literal_eval(str(val))
        initial_rt = float(parsed[0]) * 1000
        if len(parsed) > 1:
            diffs = [parsed[i + 1] - parsed[i] for i in range(len(parsed) - 1)]
            mean_itt = float(np.mean(diffs)) * 1000
        else:
            mean_itt = 0.0
        return (initial_rt, mean_itt)
    except (ValueError, SyntaxError, IndexError):
        return (np.nan, np.nan)


def clean_phone_data(df):
    df = df.copy()
    df["Completed"] = df["Completed"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    for col in ["InitialResponseTime(ms)", "SuccessRate(%)", "HitRate(%)",
                "FalseAlarms", "AvgInterTargetTime(ms)", "Level", "FinalScore"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["participant_id", "Timestamp"]).reset_index(drop=True)
    df["is_retry"] = df.duplicated(subset=["participant_id", "Level"], keep="first")
    return df


def aggregate_phone_participant(df):
    records = []
    for pid, grp in df.groupby("participant_id"):
        completed = grp[grp["Completed"] == True]
        rec = {
            "participant_id": pid,
            "target_count": grp["target_count"].iloc[0],
            "device": "phone",
            "mean_rt_ms": completed["InitialResponseTime(ms)"].mean(),
            "median_rt_ms": completed["InitialResponseTime(ms)"].median(),
            "sd_rt_ms": completed["InitialResponseTime(ms)"].std(),
            "mean_success_rate": grp["SuccessRate(%)"].mean(),
            "mean_hit_rate": grp["HitRate(%)"].mean(),
            "mean_false_alarms": grp["FalseAlarms"].mean(),
            "total_false_alarms": grp["FalseAlarms"].sum(),
            "n_levels_attempted": len(grp),
            "n_levels_completed": len(completed),
            "n_retries": grp["is_retry"].sum(),
            "max_level_reached": completed["Level"].max() if len(completed) > 0 else 0,
        }
        if grp["target_count"].iloc[0] == "multiple":
            rec["mean_inter_target_ms"] = completed["AvgInterTargetTime(ms)"].mean()
        records.append(rec)
    return pd.DataFrame(records)


def aggregate_lab_participant(df):
    records = []
    for pid, grp in df.groupby("participant_id"):
        rec = {
            "participant_id": pid,
            "target_count": grp["target_count"].iloc[0],
            "device": "lab",
            "mean_rt_ms": grp["response_time_ms"].mean(),
            "median_rt_ms": grp["response_time_ms"].median(),
            "sd_rt_ms": grp["response_time_ms"].std(),
            "n_trials": len(grp),
            "mean_rt_white_ms": grp.loc[grp["target_col"] == "white", "response_time_ms"].mean(),
            "mean_rt_red_ms": grp.loc[grp["target_col"] == "red", "response_time_ms"].mean(),
            "n_white_trials": (grp["target_col"] == "white").sum(),
            "n_red_trials": (grp["target_col"] == "red").sum(),
        }
        if "mean_inter_target_ms" in grp.columns:
            rec["mean_inter_target_ms"] = grp["mean_inter_target_ms"].mean()
        records.append(rec)
    return pd.DataFrame(records)


def build_unified_summary(agg_phone_single, agg_phone_multiple,
                          agg_lab_single, agg_lab_multiple):
    common_cols = ["participant_id", "target_count", "device",
                   "mean_rt_ms", "median_rt_ms", "sd_rt_ms"]
    frames = []
    for agg in [agg_phone_single, agg_phone_multiple, agg_lab_single, agg_lab_multiple]:
        cols = [c for c in common_cols if c in agg.columns]
        frames.append(agg[cols])
    return pd.concat(frames, ignore_index=True)


def print_separator(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def print_descriptive_tables(unified, agg_phone_single, agg_phone_multiple,
                             agg_lab_single, agg_lab_multiple,
                             lab_single_raw, lab_multiple_raw):
    print_separator("Table 1: Mean Response Time (ms) by Condition")
    stats = unified.groupby(["target_count", "device"])["mean_rt_ms"].agg(
        N="count", Mean="mean", Median="median", SD="std", Min="min", Max="max"
    ).round(1)
    print(stats.to_string())

    print_separator("Table 2: Response Time (ms) by Target Color (Lab Only)")
    lab_all = pd.concat([lab_single_raw, lab_multiple_raw], ignore_index=True)
    color_stats = lab_all.groupby(["target_count", "target_col"])["response_time_ms"].agg(
        N="count", Mean="mean", Median="median", SD="std"
    ).round(1)
    print(color_stats.to_string())

    print_separator("Table 3: Accuracy Metrics (Phone Only)")
    phone_agg = pd.concat([agg_phone_single, agg_phone_multiple], ignore_index=True)
    acc_stats = phone_agg.groupby("target_count").agg(
        N=("participant_id", "count"),
        Mean_SuccessRate=("mean_success_rate", "mean"),
        SD_SuccessRate=("mean_success_rate", "std"),
        Mean_HitRate=("mean_hit_rate", "mean"),
        SD_HitRate=("mean_hit_rate", "std"),
        Mean_FalseAlarms=("mean_false_alarms", "mean"),
        SD_FalseAlarms=("mean_false_alarms", "std"),
    ).round(2)
    print(acc_stats.to_string())

    print_separator("Table 4: Task Completion (Phone Only)")
    comp_stats = phone_agg.groupby("target_count").agg(
        N=("participant_id", "count"),
        Mean_LevelsAttempted=("n_levels_attempted", "mean"),
        Mean_LevelsCompleted=("n_levels_completed", "mean"),
        Mean_Retries=("n_retries", "mean"),
        Mean_MaxLevel=("max_level_reached", "mean"),
        SD_MaxLevel=("max_level_reached", "std"),
    ).round(2)
    print(comp_stats.to_string())

    print_separator("Table 5: Inter-Target Time in ms (Multiple Condition Only)")
    itt_records = []
    if "mean_inter_target_ms" in agg_phone_multiple.columns:
        vals = agg_phone_multiple["mean_inter_target_ms"].dropna()
        itt_records.append({
            "Device": "phone", "N": len(vals),
            "Mean": vals.mean(), "Median": vals.median(), "SD": vals.std()
        })
    if "mean_inter_target_ms" in agg_lab_multiple.columns:
        vals = agg_lab_multiple["mean_inter_target_ms"].dropna()
        itt_records.append({
            "Device": "lab", "N": len(vals),
            "Mean": vals.mean(), "Median": vals.median(), "SD": vals.std()
        })
    if itt_records:
        print(pd.DataFrame(itt_records).set_index("Device").round(1).to_string())


def plot_rt_boxplots(unified):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=unified, x="target_count", y="mean_rt_ms", hue="device",
                palette="Set2", ax=ax)
    sns.stripplot(data=unified, x="target_count", y="mean_rt_ms", hue="device",
                  palette="Set2", dodge=True, alpha=0.5, size=4, ax=ax,
                  legend=False)
    ax.set_xlabel("Target Count")
    ax.set_ylabel("Mean Response Time (ms)")
    ax.set_title("Response Time Distribution by Condition")
    ax.legend(title="Device")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rt_boxplot_by_condition.png"))
    plt.close(fig)


def plot_rt_bar_chart(unified):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=unified, x="target_count", y="mean_rt_ms", hue="device",
                palette="Set2", errorbar="sd", ax=ax)
    ax.set_xlabel("Target Count")
    ax.set_ylabel("Mean Response Time (ms)")
    ax.set_title("Mean Response Time by Condition (error bars = SD)")
    ax.legend(title="Device")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rt_bar_chart_by_condition.png"))
    plt.close(fig)


def plot_rt_histograms(unified):
    conditions = [
        ("single", "phone"), ("single", "lab"),
        ("multiple", "phone"), ("multiple", "lab"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (tc, dev) in zip(axes.flat, conditions):
        subset = unified[(unified["target_count"] == tc) & (unified["device"] == dev)]
        sns.histplot(subset["mean_rt_ms"], kde=True, ax=ax, color="steelblue", bins=8)
        ax.set_title(f"{tc} / {dev} (n={len(subset)})")
        ax.set_xlabel("Mean RT (ms)")
    plt.suptitle("Response Time Distributions by Condition", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rt_histograms.png"))
    plt.close(fig)


def plot_accuracy_charts(agg_phone_single, agg_phone_multiple):
    phone_agg = pd.concat([agg_phone_single, agg_phone_multiple], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.boxplot(data=phone_agg, x="target_count", y="mean_success_rate",
                palette="Pastel1", flierprops={"marker": ""}, ax=axes[0])
    sns.stripplot(data=phone_agg, x="target_count", y="mean_success_rate",
                  color="black", alpha=0.4, size=4, ax=axes[0])
    axes[0].set_ylabel("Mean Success Rate (%)")
    axes[0].set_title("Success Rate by Target Count")
    axes[0].set_xlabel("Target Count")

    sns.boxplot(data=phone_agg, x="target_count", y="mean_hit_rate",
                palette="Pastel1", flierprops={"marker": ""}, ax=axes[1])
    sns.stripplot(data=phone_agg, x="target_count", y="mean_hit_rate",
                  color="black", alpha=0.4, size=4, ax=axes[1])
    axes[1].set_ylabel("Mean Hit Rate (%)")
    axes[1].set_title("Hit Rate by Target Count")
    axes[1].set_xlabel("Target Count")

    plt.suptitle("Accuracy Metrics (Phone Data)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "accuracy_boxplot_phone.png"))
    plt.close(fig)


def plot_false_alarms(agg_phone_single, agg_phone_multiple):
    phone_agg = pd.concat([agg_phone_single, agg_phone_multiple], ignore_index=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=phone_agg, x="target_count", y="mean_false_alarms",
                palette="Pastel2", ax=ax)
    sns.stripplot(data=phone_agg, x="target_count", y="mean_false_alarms",
                  color="black", alpha=0.4, size=4, ax=ax)
    ax.set_xlabel("Target Count")
    ax.set_ylabel("Mean False Alarms per Level")
    ax.set_title("False Alarms by Target Count (Phone Data)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "false_alarms_boxplot.png"))
    plt.close(fig)


def plot_target_color_rt(lab_single_raw, lab_multiple_raw):
    lab_all = pd.concat([lab_single_raw, lab_multiple_raw], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=lab_all, x="target_count", y="response_time_ms",
                hue="target_col", palette={"white": "#cccccc", "red": "#e74c3c"}, ax=ax)
    ax.set_xlabel("Target Count")
    ax.set_ylabel("Response Time (ms)")
    ax.set_title("Response Time by Target Color (Lab Data)")
    ax.legend(title="Target Color")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rt_by_target_color_lab.png"))
    plt.close(fig)


def plot_inter_target_time(agg_phone_multiple, agg_lab_multiple):
    records = []
    if "mean_inter_target_ms" in agg_phone_multiple.columns:
        for _, row in agg_phone_multiple.iterrows():
            records.append({"Device": "phone", "Mean Inter-Target Time (ms)": row["mean_inter_target_ms"]})
    if "mean_inter_target_ms" in agg_lab_multiple.columns:
        for _, row in agg_lab_multiple.iterrows():
            records.append({"Device": "lab", "Mean Inter-Target Time (ms)": row["mean_inter_target_ms"]})
    if not records:
        return
    itt_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=itt_df, x="Device", y="Mean Inter-Target Time (ms)",
                palette="Set3", ax=ax)
    sns.stripplot(data=itt_df, x="Device", y="Mean Inter-Target Time (ms)",
                  color="black", alpha=0.4, size=4, ax=ax)
    ax.set_title("Inter-Target Time: Phone vs Lab (Multiple Targets)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "inter_target_time_multiple.png"))
    plt.close(fig)


def plot_rt_paired_lines(unified):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, tc in zip(axes, ["single", "multiple"]):
        subset = unified[unified["target_count"] == tc]
        pivot = subset.pivot(index="participant_id", columns="device", values="mean_rt_ms")
        if "phone" not in pivot.columns or "lab" not in pivot.columns:
            continue
        pivot = pivot.dropna()
        for _, row in pivot.iterrows():
            ax.plot(["phone", "lab"], [row["phone"], row["lab"]],
                    color="gray", alpha=0.4, linewidth=1)
        ax.scatter(["phone"] * len(pivot), pivot["phone"], color="steelblue", s=30, zorder=3)
        ax.scatter(["lab"] * len(pivot), pivot["lab"], color="coral", s=30, zorder=3)
        ax.set_title(f"{tc.capitalize()} Target (n={len(pivot)})")
        ax.set_ylabel("Mean Response Time (ms)")
        ax.set_xlabel("Device")
    plt.suptitle("Within-Subject Device Comparison", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rt_paired_lines.png"))
    plt.close(fig)


def print_rt_variability_table(unified):
    print_separator("Table 6: Within-Participant RT Variability — SD of RT (ms) by Condition")
    stats = unified.groupby(["target_count", "device"])["sd_rt_ms"].agg(
        N="count", Mean="mean", Median="median", SD="std", Min="min", Max="max"
    ).round(1)
    print(stats.to_string())


def plot_rt_variability(unified):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=unified, x="target_count", y="sd_rt_ms", hue="device",
                palette="Set2", ax=ax)
    sns.stripplot(data=unified, x="target_count", y="sd_rt_ms", hue="device",
                  palette="Set2", dodge=True, alpha=0.5, size=4, ax=ax,
                  legend=False)
    ax.set_xlabel("Target Count")
    ax.set_ylabel("Within-Participant SD of RT (ms)")
    ax.set_title("Response Time Variability by Condition")
    ax.legend(title="Device")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rt_variability_boxplot.png"))
    plt.close(fig)


def plot_device_consistency(unified):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, tc in zip(axes, ["single", "multiple"]):
        subset = unified[unified["target_count"] == tc]
        pivot = subset.pivot(index="participant_id", columns="device", values="mean_rt_ms")
        if "phone" not in pivot.columns or "lab" not in pivot.columns:
            continue
        pivot = pivot.dropna()
        phone_vals = pivot["phone"].values
        lab_vals = pivot["lab"].values

        ax.scatter(phone_vals, lab_vals, color="steelblue", s=40, zorder=3, edgecolors="white")

        all_vals = np.concatenate([phone_vals, lab_vals])
        lims = [all_vals.min() * 0.9, all_vals.max() * 1.1]
        ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="identity line")

        if len(pivot) >= 3:
            r, p = pearsonr(phone_vals, lab_vals)
            slope, intercept = np.polyfit(phone_vals, lab_vals, 1)
            x_line = np.linspace(phone_vals.min(), phone_vals.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, "-", color="coral", linewidth=2)
            p_str = f"p={p:.3f}" if p >= .001 else "p<.001"
            ax.annotate(f"r={r:.2f}, {p_str}", xy=(0.05, 0.95), xycoords="axes fraction",
                        fontsize=10, ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            print(f"  Device consistency ({tc}): r={r:.3f}, p={p:.4f}, n={len(pivot)}")

        ax.set_xlabel("Phone Mean RT (ms)")
        ax.set_ylabel("Lab Mean RT (ms)")
        ax.set_title(f"{tc.capitalize()} Target (n={len(pivot)})")
        ax.legend(loc="lower right", fontsize=8)

    plt.suptitle("Cross-Device RT Consistency", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "device_consistency_scatter.png"))
    plt.close(fig)


def plot_speed_accuracy_tradeoff(agg_phone_single, agg_phone_multiple):
    fig, ax = plt.subplots(figsize=(8, 5))

    for agg, tc, color in [(agg_phone_single, "single", "steelblue"),
                            (agg_phone_multiple, "multiple", "coral")]:
        rt = agg["mean_rt_ms"].values
        acc = agg["mean_success_rate"].values
        mask = ~(np.isnan(rt) | np.isnan(acc))
        rt, acc = rt[mask], acc[mask]

        ax.scatter(rt, acc, color=color, s=40, label=tc, edgecolors="white", zorder=3)

        if len(rt) >= 3:
            r, p = pearsonr(rt, acc)
            slope, intercept = np.polyfit(rt, acc, 1)
            x_line = np.linspace(rt.min(), rt.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, "-", color=color, linewidth=2, alpha=0.7)
            p_str = f"p={p:.3f}" if p >= .001 else "p<.001"
            print(f"  Speed-accuracy ({tc}): r={r:.3f}, p={p:.4f}, n={len(rt)}")
            y_pos = 0.95 if tc == "single" else 0.85
            ax.annotate(f"{tc}: r={r:.2f}, {p_str}", xy=(0.05, y_pos),
                        xycoords="axes fraction", fontsize=10, ha="left", va="top",
                        color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("Mean Response Time (ms)")
    ax.set_ylabel("Mean Success Rate (%)")
    ax.set_title("Speed–Accuracy Tradeoff (Phone Data)")
    ax.legend(title="Target Count")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "speed_accuracy_tradeoff.png"))
    plt.close(fig)


def plot_learning_curves(phone_single, phone_multiple, lab_single, lab_multiple):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for raw, tc, color in [(phone_single, "single", "steelblue"),
                            (phone_multiple, "multiple", "coral")]:
        first_attempts = raw[(raw["Completed"] == True) & (raw["is_retry"] == False)].copy()
        level_stats = first_attempts.groupby("Level")["InitialResponseTime(ms)"].agg(
            mean="mean", sem="sem", n="count"
        ).reset_index()
        level_stats = level_stats[level_stats["n"] >= 15]
        ax.plot(level_stats["Level"], level_stats["mean"], "-o", color=color,
                label=f"{tc} (n per level ≥ {int(level_stats['n'].min())})",
                markersize=4, linewidth=1.5)
        ax.fill_between(level_stats["Level"],
                        level_stats["mean"] - level_stats["sem"],
                        level_stats["mean"] + level_stats["sem"],
                        color=color, alpha=0.15)
    ax.set_xlabel("Phone Level")
    ax.set_ylabel("Mean Initial RT (ms)")
    ax.set_title("Phone: RT by Level (difficulty)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for raw, tc, color in [(lab_single, "single", "steelblue"),
                            (lab_multiple, "multiple", "coral")]:
        trial_stats = raw.groupby("trials.thisN")["response_time_ms"].agg(
            mean="mean", sem="sem", n="count"
        ).reset_index()
        ax.plot(trial_stats["trials.thisN"], trial_stats["mean"], "-o", color=color,
                label=f"{tc} (n={int(trial_stats['n'].iloc[0])})",
                markersize=4, linewidth=1.5)
        ax.fill_between(trial_stats["trials.thisN"],
                        trial_stats["mean"] - trial_stats["sem"],
                        trial_stats["mean"] + trial_stats["sem"],
                        color=color, alpha=0.15)
    ax.set_xlabel("Lab Trial Number")
    ax.set_ylabel("Mean RT (ms)")
    ax.set_title("Lab: RT by Trial (practice order)")
    ax.legend(fontsize=8)

    plt.suptitle("Learning / Practice Curves", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "learning_curves.png"))
    plt.close(fig)


def plot_level_difficulty(phone_single, phone_multiple):
    fig, ax = plt.subplots(figsize=(8, 5))

    for raw, tc, color in [(phone_single, "single", "steelblue"),
                            (phone_multiple, "multiple", "coral")]:
        first_attempts = raw[raw["is_retry"] == False].copy()
        level_stats = first_attempts.groupby("Level")["SuccessRate(%)"].agg(
            mean="mean", sem="sem", n="count"
        ).reset_index()
        level_stats = level_stats[level_stats["n"] >= 3]
        ax.plot(level_stats["Level"], level_stats["mean"], "-o", color=color,
                label=f"{tc}", markersize=5, linewidth=1.5)
        ax.fill_between(level_stats["Level"],
                        level_stats["mean"] - level_stats["sem"],
                        level_stats["mean"] + level_stats["sem"],
                        color=color, alpha=0.15)
        for _, row in level_stats.iterrows():
            if row["Level"] in [1, 5, 9, 13, 15] or row["n"] < 5:
                ax.annotate(f"n={int(row['n'])}", (row["Level"], row["mean"]),
                            textcoords="offset points", xytext=(0, 10),
                            fontsize=7, ha="center", color=color, alpha=0.7)

    ax.set_xlabel("Phone Level")
    ax.set_ylabel("Mean Success Rate (%)")
    ax.set_title("Level Difficulty Profile (Phone Data — First Attempts)")
    ax.legend(title="Target Count")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "level_difficulty_profile.png"))
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  DESCRIPTIVE STATISTICS: ATTENTION VALIDATION TASK")
    print("=" * 65)

    print("\nLoading data...")
    phone_single = load_phone_data("single", SINGLE_IDS)
    phone_multiple = load_phone_data("multiple", MULTIPLE_IDS)
    lab_single = load_lab_data("single", SINGLE_IDS)
    lab_multiple = load_lab_data("multiple", MULTIPLE_IDS)
    print(f"  Phone single : {len(phone_single)} rows from {phone_single['participant_id'].nunique()} participants")
    print(f"  Phone multiple: {len(phone_multiple)} rows from {phone_multiple['participant_id'].nunique()} participants")
    print(f"  Lab single    : {len(lab_single)} rows from {lab_single['participant_id'].nunique()} participants")
    print(f"  Lab multiple  : {len(lab_multiple)} rows from {lab_multiple['participant_id'].nunique()} participants")

    print("\nCleaning data...")
    phone_single = clean_phone_data(phone_single)
    phone_multiple = clean_phone_data(phone_multiple)

    print("Aggregating per participant...")
    agg_phone_single = aggregate_phone_participant(phone_single)
    agg_phone_multiple = aggregate_phone_participant(phone_multiple)
    agg_lab_single = aggregate_lab_participant(lab_single)
    agg_lab_multiple = aggregate_lab_participant(lab_multiple)

    unified = build_unified_summary(agg_phone_single, agg_phone_multiple,
                                    agg_lab_single, agg_lab_multiple)
    print(f"Unified summary: {len(unified)} rows "
          f"({unified['participant_id'].nunique()} participants x {unified['device'].nunique()} devices)")

    print_descriptive_tables(unified, agg_phone_single, agg_phone_multiple,
                            agg_lab_single, agg_lab_multiple,
                            lab_single, lab_multiple)
    print_rt_variability_table(unified)

    print(f"\n{'=' * 65}")
    print("  Generating plots...")
    print(f"{'=' * 65}")
    plot_rt_boxplots(unified)
    print("  [1/13] rt_boxplot_by_condition.png")
    plot_rt_bar_chart(unified)
    print("  [2/13] rt_bar_chart_by_condition.png")
    plot_rt_histograms(unified)
    print("  [3/13] rt_histograms.png")
    plot_accuracy_charts(agg_phone_single, agg_phone_multiple)
    print("  [4/13] accuracy_boxplot_phone.png")
    plot_false_alarms(agg_phone_single, agg_phone_multiple)
    print("  [5/13] false_alarms_boxplot.png")
    plot_target_color_rt(lab_single, lab_multiple)
    print("  [6/13] rt_by_target_color_lab.png")
    plot_inter_target_time(agg_phone_multiple, agg_lab_multiple)
    print("  [7/13] inter_target_time_multiple.png")
    plot_rt_paired_lines(unified)
    print("  [8/13] rt_paired_lines.png")
    plot_rt_variability(unified)
    print("  [9/13] rt_variability_boxplot.png")
    plot_device_consistency(unified)
    print("  [10/13] device_consistency_scatter.png")
    plot_speed_accuracy_tradeoff(agg_phone_single, agg_phone_multiple)
    print("  [11/13] speed_accuracy_tradeoff.png")
    plot_learning_curves(phone_single, phone_multiple, lab_single, lab_multiple)
    print("  [12/13] learning_curves.png")
    plot_level_difficulty(phone_single, phone_multiple)
    print("  [13/13] level_difficulty_profile.png")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
