import sys
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from descriptive_statistics import (
    load_phone_data, load_lab_data, clean_phone_data,
    aggregate_phone_participant, aggregate_lab_participant,
    build_unified_summary, apply_rt_outlier_filter,
    SINGLE_IDS, MULTIPLE_IDS, OUTPUT_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})

ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / 4
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


def load_all_data():
    phone_single = clean_phone_data(load_phone_data("single", SINGLE_IDS))
    phone_multiple = clean_phone_data(load_phone_data("multiple", MULTIPLE_IDS))
    lab_single = load_lab_data("single", SINGLE_IDS)
    lab_multiple = load_lab_data("multiple", MULTIPLE_IDS)

    agg_phone_single = aggregate_phone_participant(phone_single)
    agg_phone_multiple = aggregate_phone_participant(phone_multiple)
    agg_lab_single = aggregate_lab_participant(lab_single)
    agg_lab_multiple = aggregate_lab_participant(lab_multiple)

    (agg_phone_single, agg_phone_multiple,
     agg_lab_single, agg_lab_multiple,
     outlier_summary, outlier_rows) = apply_rt_outlier_filter(
        agg_phone_single, agg_phone_multiple, agg_lab_single, agg_lab_multiple
    )

    unified = build_unified_summary(
        agg_phone_single, agg_phone_multiple,
        agg_lab_single, agg_lab_multiple,
    )

    return {
        "unified": unified,
        "agg_phone_single": agg_phone_single,
        "agg_phone_multiple": agg_phone_multiple,
        "agg_lab_single": agg_lab_single,
        "agg_lab_multiple": agg_lab_multiple,
        "outlier_summary": outlier_summary,
        "outlier_rows": outlier_rows,
    }


def run_assumption_checks(unified):
    section("SECTION 1: ASSUMPTION CHECKS")

    subsection("Normality — Raw Mean RT (Shapiro-Wilk)")
    norm_rows = []
    for tc in ["single", "multiple"]:
        for dev in ["phone", "lab"]:
            vals = unified[(unified["target_count"] == tc) & (unified["device"] == dev)]["mean_rt_ms"].dropna()
            w, p = stats.shapiro(vals)
            sk = stats.skew(vals)
            norm_rows.append({
                "Condition": f"{tc}/{dev}",
                "n": len(vals),
                "W": round(w, 4),
                "p": round(p, 4),
                "Skewness": round(sk, 3),
                "Normal?": "Yes" if p >= ALPHA else "NO",
            })
    norm_df = pd.DataFrame(norm_rows)
    print(norm_df.to_string(index=False))

    unified["log_mean_rt_ms"] = np.log(unified["mean_rt_ms"])

    subsection("Normality — Log-Transformed Mean RT (Shapiro-Wilk)")
    log_norm_rows = []
    for tc in ["single", "multiple"]:
        for dev in ["phone", "lab"]:
            vals = unified[(unified["target_count"] == tc) & (unified["device"] == dev)]["log_mean_rt_ms"].dropna()
            w, p = stats.shapiro(vals)
            sk = stats.skew(vals)
            log_norm_rows.append({
                "Condition": f"{tc}/{dev}",
                "n": len(vals),
                "W": round(w, 4),
                "p": round(p, 4),
                "Skewness": round(sk, 3),
                "Normal?": "Yes" if p >= ALPHA else "NO",
            })
    log_norm_df = pd.DataFrame(log_norm_rows)
    print(log_norm_df.to_string(index=False))

    subsection("Sphericity")
    print("  Within-subjects factor 'device' has only 2 levels (phone, lab).")
    print("  Sphericity is automatically satisfied — no correction needed.")

    subsection("Homogeneity of Variances (Levene's Test)")

    groups_raw = []
    labels = []
    for tc in ["single", "multiple"]:
        for dev in ["phone", "lab"]:
            vals = unified[(unified["target_count"] == tc) & (unified["device"] == dev)]["mean_rt_ms"].dropna()
            groups_raw.append(vals.values)
            labels.append(f"{tc}/{dev}")
    lev_stat, lev_p = stats.levene(*groups_raw)

    groups_log = []
    for tc in ["single", "multiple"]:
        for dev in ["phone", "lab"]:
            vals = unified[(unified["target_count"] == tc) & (unified["device"] == dev)]["log_mean_rt_ms"].dropna()
            groups_log.append(vals.values)
    lev_log_stat, lev_log_p = stats.levene(*groups_log)

    lev_rows = []
    for lbl, g_raw, g_log in zip(labels, groups_raw, groups_log):
        lev_rows.append({
            "Condition": lbl,
            "Var(raw)": round(np.var(g_raw, ddof=1), 1),
            "Var(log)": round(np.var(g_log, ddof=1), 4),
        })
    print(pd.DataFrame(lev_rows).to_string(index=False))
    print(f"\n  Levene's (raw RT): F={lev_stat:.4f}, p={lev_p:.4f}  {sig_marker(lev_p)}")
    print(f"  Levene's (log RT): F={lev_log_stat:.4f}, p={lev_log_p:.4f}  {sig_marker(lev_log_p)}")

    subsection("Assumption Summary")
    raw_normal_count = sum(1 for r in norm_rows if r["Normal?"] == "Yes")
    log_normal_count = sum(1 for r in log_norm_rows if r["Normal?"] == "Yes")
    print(f"  Raw RT:  {raw_normal_count}/4 conditions pass normality; "
          f"Levene's p={lev_p:.4f} {'(homogeneous)' if lev_p >= ALPHA else '(HETEROGENEOUS)'}")
    print(f"  Log RT:  {log_normal_count}/4 conditions pass normality; "
          f"Levene's p={lev_log_p:.4f} {'(homogeneous)' if lev_log_p >= ALPHA else '(HETEROGENEOUS)'}")
    print(f"\n  -> Using log-transformed RT as primary DV for mixed ANOVA.")
    print(f"  -> Raw RT results reported as robustness check.")

    return unified


def run_mixed_anova(unified):
    section("SECTION 2: PRIMARY ANALYSIS — MIXED ANOVA ON RESPONSE TIME")

    anova_df = unified[["participant_id", "target_count", "device", "mean_rt_ms", "log_mean_rt_ms"]].dropna()

    subsection("Mixed ANOVA — Log Mean RT (primary)")
    aov_log = pg.mixed_anova(
        data=anova_df,
        dv="log_mean_rt_ms",
        between="target_count",
        within="device",
        subject="participant_id",
    )
    aov_log = aov_log.round(4)
    print(aov_log.to_string(index=False))

    for _, row in aov_log.iterrows():
        src = row["Source"]
        p = row["p_unc"]
        eta = row.get("np2", np.nan)
        print(f"\n  {src}: F({row['DF1']:.0f},{row['DF2']:.0f}) = {row['F']:.3f}, "
              f"p = {p:.4f} {sig_marker(p)}, partial eta^2 = {eta:.4f}")

    subsection("Mixed ANOVA — Raw Mean RT (robustness check)")
    aov_raw = pg.mixed_anova(
        data=anova_df,
        dv="mean_rt_ms",
        between="target_count",
        within="device",
        subject="participant_id",
    )
    aov_raw = aov_raw.round(4)
    print(aov_raw.to_string(index=False))

    for _, row in aov_raw.iterrows():
        src = row["Source"]
        p = row["p_unc"]
        eta = row.get("np2", np.nan)
        print(f"\n  {src}: F({row['DF1']:.0f},{row['DF2']:.0f}) = {row['F']:.3f}, "
              f"p = {p:.4f} {sig_marker(p)}, partial eta^2 = {eta:.4f}")

    subsection("Log vs Raw ANOVA Comparison")
    for i in range(len(aov_log)):
        src = aov_log.iloc[i]["Source"]
        p_log = aov_log.iloc[i]["p_unc"]
        p_raw = aov_raw.iloc[i]["p_unc"]
        agree = (p_log < ALPHA) == (p_raw < ALPHA)
        print(f"  {src}: log p={p_log:.4f}, raw p={p_raw:.4f}  "
              f"{'AGREE' if agree else 'DISAGREE'}")

    return aov_log, aov_raw


def run_simple_effects(unified):
    section("SECTION 3: FOLLOW-UP — SIMPLE EFFECTS (Bonferroni-corrected alpha = 0.0125)")

    results = []

    subsection("Device effect within each target_count (paired t-test + Wilcoxon)")

    for tc in ["single", "multiple"]:
        phone_vals = unified[(unified["target_count"] == tc) & (unified["device"] == "phone")].set_index("participant_id")["mean_rt_ms"]
        lab_vals = unified[(unified["target_count"] == tc) & (unified["device"] == "lab")].set_index("participant_id")["mean_rt_ms"]
        common = phone_vals.index.intersection(lab_vals.index)
        phone = phone_vals.loc[common].values
        lab = lab_vals.loc[common].values
        diff = phone - lab
        n = len(common)

        t_stat, t_p = stats.ttest_rel(phone, lab)

        dz = diff.mean() / diff.std()

        w_stat, w_p = stats.wilcoxon(phone, lab)

        print(f"  {tc.upper()} target (n={n}):")
        print(f"    Mean phone = {phone.mean():.1f} ms, Mean lab = {lab.mean():.1f} ms")
        print(f"    Mean diff  = {diff.mean():.1f} ms (SD = {diff.std():.1f})")
        print(f"    Paired t-test: t({n-1}) = {t_stat:.3f}, p = {t_p:.6f}  "
              f"{sig_marker(t_p, BONFERRONI_ALPHA)}")
        print(f"    Cohen's dz = {dz:.3f}")
        print(f"    Wilcoxon: W = {w_stat:.1f}, p = {w_p:.6f}  "
              f"{sig_marker(w_p, BONFERRONI_ALPHA)}")
        print()

        results.append({
            "Comparison": f"Device effect ({tc})",
            "Type": "paired",
            "n": n,
            "Test": "Paired t",
            "Statistic": round(t_stat, 3),
            "p": round(t_p, 6),
            "Effect size": f"dz={dz:.3f}",
            "Sig (Bonf)": sig_marker(t_p, BONFERRONI_ALPHA),
        })
        results.append({
            "Comparison": f"Device effect ({tc})",
            "Type": "paired",
            "n": n,
            "Test": "Wilcoxon",
            "Statistic": round(w_stat, 1),
            "p": round(w_p, 6),
            "Effect size": f"dz={dz:.3f}",
            "Sig (Bonf)": sig_marker(w_p, BONFERRONI_ALPHA),
        })

    subsection("Target_count effect within each device (Welch's t-test + Mann-Whitney)")

    for dev in ["phone", "lab"]:
        single_vals = unified[(unified["target_count"] == "single") & (unified["device"] == dev)]["mean_rt_ms"].values
        multiple_vals = unified[(unified["target_count"] == "multiple") & (unified["device"] == dev)]["mean_rt_ms"].values
        n1, n2 = len(single_vals), len(multiple_vals)

        t_stat, t_p = stats.ttest_ind(single_vals, multiple_vals, equal_var=False)

        pooled_sd = np.sqrt(
            ((n1 - 1) * single_vals.std(ddof=1)**2 + (n2 - 1) * multiple_vals.std(ddof=1)**2)
            / (n1 + n2 - 2)
        )
        d = (single_vals.mean() - multiple_vals.mean()) / pooled_sd

        u_stat, u_p = stats.mannwhitneyu(single_vals, multiple_vals, alternative="two-sided")
        rbc = 1 - (2 * u_stat) / (n1 * n2)

        print(f"  {dev.upper()} device (single n={n1}, multiple n={n2}):")
        print(f"    Mean single = {single_vals.mean():.1f} ms, Mean multiple = {multiple_vals.mean():.1f} ms")
        print(f"    Welch's t-test: t = {t_stat:.3f}, p = {t_p:.6f}  "
              f"{sig_marker(t_p, BONFERRONI_ALPHA)}")
        print(f"    Cohen's d = {d:.3f}")
        print(f"    Mann-Whitney: U = {u_stat:.1f}, p = {u_p:.6f}  "
              f"{sig_marker(u_p, BONFERRONI_ALPHA)}")
        print(f"    Rank-biserial r = {rbc:.3f}")
        print()

        results.append({
            "Comparison": f"Target_count effect ({dev})",
            "Type": "independent",
            "n": f"{n1}+{n2}",
            "Test": "Welch's t",
            "Statistic": round(t_stat, 3),
            "p": round(t_p, 6),
            "Effect size": f"d={d:.3f}",
            "Sig (Bonf)": sig_marker(t_p, BONFERRONI_ALPHA),
        })
        results.append({
            "Comparison": f"Target_count effect ({dev})",
            "Type": "independent",
            "n": f"{n1}+{n2}",
            "Test": "Mann-Whitney",
            "Statistic": round(u_stat, 1),
            "p": round(u_p, 6),
            "Effect size": f"r={rbc:.3f}",
            "Sig (Bonf)": sig_marker(u_p, BONFERRONI_ALPHA),
        })

    return results


def run_target_color_analysis(data):
    section("SECTION 4: LAB TARGET COLOR ANALYSIS")

    results = []

    for tc, agg_lab in [("single", data["agg_lab_single"]), ("multiple", data["agg_lab_multiple"])]:
        white_rt = agg_lab["mean_rt_white_ms"].dropna().values
        red_rt = agg_lab["mean_rt_red_ms"].dropna().values
        n = min(len(white_rt), len(red_rt))
        white_rt = white_rt[:n]
        red_rt = red_rt[:n]

        diff = white_rt - red_rt
        t_stat, t_p = stats.ttest_rel(white_rt, red_rt)
        dz = diff.mean() / diff.std() if diff.std() > 0 else 0.0
        w_stat, w_p = stats.wilcoxon(white_rt, red_rt)

        print(f"  {tc.upper()} target (n={n}):")
        print(f"    Mean white = {white_rt.mean():.1f} ms, Mean red = {red_rt.mean():.1f} ms")
        print(f"    Mean diff (white - red) = {diff.mean():.1f} ms (SD = {diff.std():.1f})")
        print(f"    Paired t-test: t({n-1}) = {t_stat:.3f}, p = {t_p:.4f}  {sig_marker(t_p)}")
        print(f"    Cohen's dz = {dz:.3f}")
        print(f"    Wilcoxon: W = {w_stat:.1f}, p = {w_p:.4f}  {sig_marker(w_p)}")
        print()

        results.append({
            "Comparison": f"White vs Red ({tc})",
            "Type": "paired",
            "n": n,
            "Test": "Paired t",
            "Statistic": round(t_stat, 3),
            "p": round(t_p, 4),
            "Effect size": f"dz={dz:.3f}",
            "Sig": sig_marker(t_p),
        })

    return results


def run_nonparametric_checks(unified):
    section("SECTION 5: NON-PARAMETRIC ROBUSTNESS CHECKS")

    rows = []

    for tc in ["single", "multiple"]:
        phone_vals = unified[(unified["target_count"] == tc) & (unified["device"] == "phone")].set_index("participant_id")["mean_rt_ms"]
        lab_vals = unified[(unified["target_count"] == tc) & (unified["device"] == "lab")].set_index("participant_id")["mean_rt_ms"]
        common = phone_vals.index.intersection(lab_vals.index)
        phone = phone_vals.loc[common].values
        lab = lab_vals.loc[common].values

        t_stat, t_p = stats.ttest_rel(phone, lab)
        w_stat, w_p = stats.wilcoxon(phone, lab)

        rows.append({
            "Comparison": f"Device ({tc})",
            "Parametric": f"t={t_stat:.3f}, p={t_p:.4f} {sig_marker(t_p)}",
            "Non-parametric": f"W={w_stat:.1f}, p={w_p:.4f} {sig_marker(w_p)}",
            "Agree?": "Yes" if (t_p < ALPHA) == (w_p < ALPHA) else "NO",
        })

    for dev in ["phone", "lab"]:
        single = unified[(unified["target_count"] == "single") & (unified["device"] == dev)]["mean_rt_ms"].values
        multiple = unified[(unified["target_count"] == "multiple") & (unified["device"] == dev)]["mean_rt_ms"].values

        t_stat, t_p = stats.ttest_ind(single, multiple, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(single, multiple, alternative="two-sided")

        rows.append({
            "Comparison": f"Target count ({dev})",
            "Parametric": f"t={t_stat:.3f}, p={t_p:.4f} {sig_marker(t_p)}",
            "Non-parametric": f"U={u_stat:.1f}, p={u_p:.4f} {sig_marker(u_p)}",
            "Agree?": "Yes" if (t_p < ALPHA) == (u_p < ALPHA) else "NO",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    all_agree = all(r["Agree?"] == "Yes" for r in rows)
    print(f"\n  All parametric and non-parametric tests agree: {'YES' if all_agree else 'NO'}")
    if all_agree:
        print("  -> Conclusions are robust to assumption violations.")


def run_accuracy_analysis(data):
    section("SECTION 6: ACCURACY ANALYSIS (PHONE DATA — NON-PARAMETRIC)")

    agg_single = data["agg_phone_single"]
    agg_multiple = data["agg_phone_multiple"]

    metrics = [
        ("mean_success_rate", "Success Rate (%)"),
        ("mean_hit_rate", "Hit Rate (%)"),
        ("mean_false_alarms", "False Alarms"),
    ]

    rows = []
    for col, label in metrics:
        single_vals = agg_single[col].dropna().values
        multiple_vals = agg_multiple[col].dropna().values
        n1, n2 = len(single_vals), len(multiple_vals)

        u_stat, u_p = stats.mannwhitneyu(single_vals, multiple_vals, alternative="two-sided")
        rbc = 1 - (2 * u_stat) / (n1 * n2)

        rows.append({
            "Metric": label,
            "Single (M +/- SD)": f"{single_vals.mean():.2f} +/- {single_vals.std(ddof=1):.2f}",
            "Multiple (M +/- SD)": f"{multiple_vals.mean():.2f} +/- {multiple_vals.std(ddof=1):.2f}",
            "U": round(u_stat, 1),
            "p": round(u_p, 4),
            "r (rank-bis)": round(rbc, 3),
            "Sig": sig_marker(u_p),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def print_results_summary(aov_log, simple_effects_results, color_results):
    section("SECTION 7: RESULTS SUMMARY")

    rows = []

    for _, row in aov_log.iterrows():
        rows.append({
            "Analysis": "Mixed ANOVA (log RT)",
            "Effect": row["Source"],
            "Test": "F-test",
            "Statistic": f"F({row['DF1']:.0f},{row['DF2']:.0f})={row['F']:.3f}",
            "p": f"{row['p_unc']:.4f}",
            "Effect Size": f"np2={row['np2']:.4f}",
            "Sig": sig_marker(row["p_unc"]),
        })

    for r in simple_effects_results:
        rows.append({
            "Analysis": "Simple Effects",
            "Effect": r["Comparison"],
            "Test": r["Test"],
            "Statistic": str(r["Statistic"]),
            "p": str(r["p"]),
            "Effect Size": r["Effect size"],
            "Sig": r["Sig (Bonf)"],
        })

    for r in color_results:
        rows.append({
            "Analysis": "Target Color",
            "Effect": r["Comparison"],
            "Test": r["Test"],
            "Statistic": str(r["Statistic"]),
            "p": str(r["p"]),
            "Effect Size": r["Effect size"],
            "Sig": r["Sig"],
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 140)
    print(df.to_string(index=False))


def plot_interaction(unified):
    section("SECTION 8: INTERACTION PLOT")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = unified.groupby(["device", "target_count"])["mean_rt_ms"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = summary["se"] * 1.96

    fig, ax = plt.subplots(figsize=(7, 5))

    device_order = ["phone", "lab"]
    colors = {"single": "#2196F3", "multiple": "#FF5722"}
    markers = {"single": "o", "multiple": "s"}

    for tc in ["single", "multiple"]:
        sub = summary[summary["target_count"] == tc].set_index("device").loc[device_order]
        ax.errorbar(
            x=device_order,
            y=sub["mean"].values,
            yerr=sub["ci95"].values,
            label=f"{tc} target",
            marker=markers[tc],
            markersize=8,
            linewidth=2,
            color=colors[tc],
            capsize=5,
            capthick=1.5,
        )

    ax.set_xlabel("Device", fontsize=12)
    ax.set_ylabel("Mean Response Time (ms)", fontsize=12)
    ax.set_title("Interaction: Device x Target Count on Response Time", fontsize=13)
    ax.legend(title="Target Count", fontsize=10)

    for tc in ["single", "multiple"]:
        for dev in device_order:
            sub = summary[(summary["target_count"] == tc) & (summary["device"] == dev)]
            n = int(sub["count"].values[0])
            y = sub["mean"].values[0] + sub["ci95"].values[0]
            ax.annotate(f"n={n}", xy=(dev, y), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "interaction_plot_rt.png")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def _hist_with_normal(ax, vals, color, title=None, xlabel=None, ylabel=None):
    vals = np.asarray(vals)
    ax.hist(vals, bins=12, color=color, alpha=0.55, edgecolor="white",
            density=True, label="Data")

    xs = np.linspace(vals.min(), vals.max(), 300)
    kde = stats.gaussian_kde(vals)
    ax.plot(xs, kde(xs), color=color, linewidth=2, label="KDE")

    mu, sigma = vals.mean(), vals.std()
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma),
            color="black", linewidth=1.8, linestyle="--", label="Normal ref")

    sk = stats.skew(vals)
    w, p = stats.shapiro(vals)
    sig = "p<.001" if p < 0.001 else f"p={p:.3f}"
    norm_color = "green" if p >= ALPHA else "#cc0000"
    status = "normal" if p >= ALPHA else "NOT normal"
    ax.annotate(
        f"skew = {sk:.2f}\nSW W={w:.3f}, {sig}\n({status})",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=8,
        color=norm_color,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    if title:
        ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


def _qqplot(ax, vals, color, title=None):
    vals = np.asarray(vals)
    (osm, osr), (slope, intercept, _) = stats.probplot(vals, dist="norm")
    ax.scatter(osm, osr, color=color, s=25, alpha=0.85, zorder=3)
    ref_x = np.array([osm.min(), osm.max()])
    ax.plot(ref_x, slope * ref_x + intercept,
            color="black", linewidth=1.4, linestyle="--")
    ax.set_xlabel("Theoretical quantiles", fontsize=8)
    ax.set_ylabel("Sample quantiles", fontsize=8)
    w, p = stats.shapiro(vals)
    sig = "p<.001" if p < 0.001 else f"p={p:.3f}"
    norm_color = "green" if p >= ALPHA else "#cc0000"
    ax.annotate(
        f"SW W={w:.3f}\n{sig}",
        xy=(0.03, 0.97), xycoords="axes fraction",
        ha="left", va="top", fontsize=8,
        color=norm_color,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)


def plot_log_transform_distributions(unified):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lab_single_raw = load_lab_data("single", SINGLE_IDS)
    lab_multiple_raw = load_lab_data("multiple", MULTIPLE_IDS)
    trial_rt = pd.concat([lab_single_raw, lab_multiple_raw])["response_time_ms"].dropna()
    trial_log_rt = np.log(trial_rt)

    mean_rt = unified["mean_rt_ms"].dropna()
    mean_log_rt = unified["log_mean_rt_ms"].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(
        "Effect of Log-Transform on RT Distributions\n"
        "(dashed line = what a perfect normal would look like)",
        fontsize=12,
    )

    _hist_with_normal(
        axes[0, 0], trial_rt, "#5B9BD5",
        title=f"Trial-level RT — Raw  (n={len(trial_rt)})",
        xlabel="Response Time (ms)", ylabel="Density",
    )
    _hist_with_normal(
        axes[0, 1], trial_log_rt, "#70AD47",
        title=f"Trial-level RT — Log  (n={len(trial_log_rt)})",
        xlabel="log(Response Time)", ylabel="Density",
    )
    _hist_with_normal(
        axes[1, 0], mean_rt, "#5B9BD5",
        title=f"Per-participant mean RT — Raw  (n={len(mean_rt)}, all conditions pooled)",
        xlabel="Mean Response Time (ms)", ylabel="Density",
    )
    _hist_with_normal(
        axes[1, 1], mean_log_rt, "#70AD47",
        title=f"Per-participant mean RT — Log  (n={len(mean_log_rt)})",
        xlabel="log(Mean Response Time)", ylabel="Density",
    )

    axes[0, 0].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, "log_transform_histograms.png")
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out1}")

    conditions = [
        ("single", "phone", "Single / Phone"),
        ("single", "lab",   "Single / Lab"),
        ("multiple", "phone", "Multiple / Phone"),
        ("multiple", "lab",   "Multiple / Lab"),
    ]

    fig2, axes2 = plt.subplots(2, 4, figsize=(14, 6))
    fig2.suptitle(
        "Q-Q Plots: Per-Participant Mean RT vs. Normal\n"
        "Row 1 = Raw RT  |  Row 2 = Log RT  (points on diagonal = perfectly normal)",
        fontsize=11,
    )

    for col_idx, (tc, dev, label) in enumerate(conditions):
        vals_raw = unified[
            (unified["target_count"] == tc) & (unified["device"] == dev)
        ]["mean_rt_ms"].dropna().values
        vals_log = unified[
            (unified["target_count"] == tc) & (unified["device"] == dev)
        ]["log_mean_rt_ms"].dropna().values

        _qqplot(axes2[0, col_idx], vals_raw, "#5B9BD5",
                title=f"{label}\n(Raw, n={len(vals_raw)})")
        _qqplot(axes2[1, col_idx], vals_log, "#70AD47",
                title=f"{label}\n(Log, n={len(vals_log)})")

    axes2[0, 0].set_ylabel("Sample quantiles\n(Raw RT)", fontsize=9)
    axes2[1, 0].set_ylabel("Sample quantiles\n(Log RT)", fontsize=9)

    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "log_transform_qqplots.png")
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2}")


def run_permutation_tests(unified):
    section("SECTION 9: PERMUTATION TESTS (RAW RT, distribution-free)")

    rng = np.random.default_rng(42)
    perm_results = []

    paired_data = {}
    for tc in ["single", "multiple"]:
        phone_vals = unified[(unified["target_count"] == tc) & (unified["device"] == "phone")].set_index("participant_id")["mean_rt_ms"]
        lab_vals = unified[(unified["target_count"] == tc) & (unified["device"] == "lab")].set_index("participant_id")["mean_rt_ms"]
        common = phone_vals.index.intersection(lab_vals.index)
        paired_data[tc] = {
            "phone": phone_vals.loc[common].values,
            "lab": lab_vals.loc[common].values,
            "diff": phone_vals.loc[common].values - lab_vals.loc[common].values,
        }

    subsection("Within-subjects: Device effect (sign-flip permutation)")

    for tc in ["single", "multiple"]:
        diff = paired_data[tc]["diff"]
        n = len(diff)
        observed = diff.mean() / (diff.std(ddof=1) / np.sqrt(n))

        null_dist = np.empty(N_PERMUTATIONS)
        for i in range(N_PERMUTATIONS):
            signs = rng.choice([-1, 1], size=n)
            flipped = signs * diff
            null_dist[i] = flipped.mean() / (flipped.std(ddof=1) / np.sqrt(n))

        p_perm = np.mean(np.abs(null_dist) >= np.abs(observed))

        p_str = f"p < 0.0001" if p_perm == 0 else f"p = {p_perm:.4f}"
        print(f"  Device effect ({tc}, n={n}):")
        print(f"    Observed t = {observed:.3f}")
        print(f"    Permutation {p_str}  {sig_marker(p_perm if p_perm > 0 else 0.00001, BONFERRONI_ALPHA)}")

        perm_results.append({
            "label": f"Device effect ({tc})",
            "observed": observed,
            "p_perm": p_perm,
            "null_distribution": null_dist,
            "n_permutations": N_PERMUTATIONS,
            "test_type": "within",
        })

    subsection("Comparison: Parametric vs Rank-based vs Permutation")

    comp_rows = []
    for tc in ["single", "multiple"]:
        phone = paired_data[tc]["phone"]
        lab = paired_data[tc]["lab"]
        _, t_p = stats.ttest_rel(phone, lab)
        _, w_p = stats.wilcoxon(phone, lab)
        pr = next(r for r in perm_results if r["label"] == f"Device effect ({tc})")
        p_perm = pr["p_perm"]
        alpha = BONFERRONI_ALPHA

        t_sig = t_p < alpha
        w_sig = w_p < alpha
        perm_sig = (p_perm if p_perm > 0 else 0.00001) < alpha
        agree = t_sig == w_sig == perm_sig

        comp_rows.append({
            "Comparison": f"Device ({tc})",
            "Parametric p": f"{t_p:.4f} {sig_marker(t_p, alpha)}",
            "Rank-based p": f"{w_p:.4f} {sig_marker(w_p, alpha)}",
            "Permutation p": f"{'<0.0001' if p_perm == 0 else f'{p_perm:.4f}'} {sig_marker(p_perm if p_perm > 0 else 0.00001, alpha)}",
            "All agree?": "Yes" if agree else "NO",
        })

    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))

    all_agree = all(r["All agree?"] == "Yes" for r in comp_rows)
    print(f"\n  All three methods agree on every comparison: {'YES' if all_agree else 'NO'}")
    if all_agree:
        print("  -> Permutation tests confirm all parametric and rank-based conclusions.")
        print("  -> Results are robust regardless of distributional assumptions.")

    return perm_results


def plot_permutation_null_distributions(perm_results):
    section("PERMUTATION NULL DISTRIBUTION PLOTS")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Permutation Null Distributions ({N_PERMUTATIONS:,} permutations)\n"
        "Red dashed = observed t  |  Orange dash-dot = critical values (2.5th / 97.5th percentile)",
        fontsize=12,
    )

    for ax, pr in zip(axes[:2], perm_results):
        null = pr["null_distribution"]
        obs = pr["observed"]
        p = pr["p_perm"]

        ax.hist(null, bins=50, color="#4A90D9", alpha=0.7, edgecolor="white", density=True)
        ax.axvline(obs, color="red", linewidth=2, linestyle="--", label=f"Observed = {obs:.3f}")

        tail_pct = BONFERRONI_ALPHA / 2 * 100
        crit_lower = np.percentile(null, tail_pct)
        crit_upper = np.percentile(null, 100 - tail_pct)
        ax.axvline(crit_lower, color="orange", linewidth=1.5, linestyle="-.",
                   label=f"{tail_pct:.2f}% = {crit_lower:.2f}")
        ax.axvline(crit_upper, color="orange", linewidth=1.5, linestyle="-.",
                   label=f"{100-tail_pct:.2f}% = {crit_upper:.2f}")

        p_str = "p < 0.0001" if p == 0 else f"p = {p:.4f}"
        sig = (p if p > 0 else 0.00001) < BONFERRONI_ALPHA
        box_color = "#ffcccc" if sig else "#e0e0e0"
        text_color = "red" if sig else "gray"
        ax.annotate(
            f"{p_str}\n{'Significant' if sig else 'Not significant'}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=9, color=text_color,
            bbox=dict(boxstyle="round,pad=0.3", fc=box_color, alpha=0.9),
        )

        ax.set_title(pr["label"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Test statistic (t-value)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

    ax_text = axes[2]
    ax_text.axis("off")
    summary_lines = ["Permutation Test Summary", "=" * 30, ""]
    for pr in perm_results:
        p = pr["p_perm"]
        p_str = "p < 0.0001" if p == 0 else f"p = {p:.4f}"
        sig = (p if p > 0 else 0.00001) < BONFERRONI_ALPHA
        marker = "**" if sig else "  "
        summary_lines.append(f"{marker} {pr['label']}")
        summary_lines.append(f"   {p_str}")
        summary_lines.append("")
    summary_lines.append(f"N permutations = {N_PERMUTATIONS:,}")
    summary_lines.append(f"Alpha (Bonferroni) = {BONFERRONI_ALPHA}")
    ax_text.text(
        0.05, 0.95, "\n".join(summary_lines),
        transform=ax_text.transAxes,
        fontsize=9, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", alpha=0.9),
    )

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "permutation_null_distributions.png")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    print("=" * 70)
    print("  INFERENTIAL STATISTICS: ATTENTION VALIDATION TASK")
    print("=" * 70)

    print("\nLoading data...")
    data = load_all_data()
    unified = data["unified"]
    outlier_summary = data.get("outlier_summary")
    outlier_rows = data.get("outlier_rows")
    if outlier_summary is not None:
        print("  Participant RT outlier removal (IQR rule, 1.5x):")
        outlier_summary_print = outlier_summary.copy()
        outlier_summary_print["Lower"] = outlier_summary_print["Lower"].round(1)
        outlier_summary_print["Upper"] = outlier_summary_print["Upper"].round(1)
        print(outlier_summary_print.to_string(index=False))
        if isinstance(outlier_rows, pd.DataFrame) and len(outlier_rows) > 0:
            removed = outlier_rows.copy()
            removed["mean_rt_ms"] = removed["mean_rt_ms"].round(1)
            removed = removed.sort_values(["target_count", "device", "participant_id"])
            print("  Removed participants:")
            print(removed.to_string(index=False))

    print(f"  {len(unified)} rows, {unified['participant_id'].nunique()} participants")

    unified = run_assumption_checks(unified)
    data["unified"] = unified

    section("SECTION 1b: LOG-TRANSFORM DISTRIBUTION PLOTS")
    plot_log_transform_distributions(unified)

    aov_log, aov_raw = run_mixed_anova(unified)

    simple_effects_results = run_simple_effects(unified)

    color_results = run_target_color_analysis(data)

    run_nonparametric_checks(unified)

    run_accuracy_analysis(data)

    print_results_summary(aov_log, simple_effects_results, color_results)

    plot_interaction(unified)

    perm_results = run_permutation_tests(unified)
    plot_permutation_null_distributions(perm_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
