import sys
sys.path.insert(0, r"C:\Users\prajwalj\Desktop\brsm_project")
from descriptive_statistics import *
import scipy.stats as stats

phone_single = load_phone_data("single", SINGLE_IDS)
phone_multiple = load_phone_data("multiple", MULTIPLE_IDS)
lab_single = load_lab_data("single", SINGLE_IDS)
lab_multiple = load_lab_data("multiple", MULTIPLE_IDS)

phone_single = clean_phone_data(phone_single)
phone_multiple = clean_phone_data(phone_multiple)

agg_phone_single = aggregate_phone_participant(phone_single)
agg_phone_multiple = aggregate_phone_participant(phone_multiple)
agg_lab_single = aggregate_lab_participant(lab_single)
agg_lab_multiple = aggregate_lab_participant(lab_multiple)

unified = build_unified_summary(agg_phone_single, agg_phone_multiple, agg_lab_single, agg_lab_multiple)

print("=" * 70)
print("1. NORMALITY OF PER-PARTICIPANT MEAN RT")
print("=" * 70)
for tc in ["single", "multiple"]:
    for dev in ["phone", "lab"]:
        subset = unified[(unified["target_count"] == tc) & (unified["device"] == dev)]["mean_rt_ms"].dropna()
        stat, p = stats.shapiro(subset)
        skew = stats.skew(subset)
        kurt = stats.kurtosis(subset)
        print(f"\n  {tc}/{dev} (n={len(subset)}):")
        print(f"    Shapiro-Wilk: W={stat:.4f}, p={p:.4f} {'*** NOT NORMAL' if p < 0.05 else '(normal)'}")
        print(f"    Skewness: {skew:.3f}, Kurtosis: {kurt:.3f}")
        print(f"    Range: {subset.min():.1f} - {subset.max():.1f}")

print("\n" + "=" * 70)
print("2. VARIANCE HOMOGENEITY (Levene's test)")
print("=" * 70)
groups = []
labels = []
for tc in ["single", "multiple"]:
    for dev in ["phone", "lab"]:
        subset = unified[(unified["target_count"] == tc) & (unified["device"] == dev)]["mean_rt_ms"].dropna()
        groups.append(subset.values)
        labels.append(f"{tc}/{dev}")
stat, p = stats.levene(*groups)
print(f"  Levene's test across all 4 groups: F={stat:.4f}, p={p:.4f}")
for lbl, g in zip(labels, groups):
    print(f"    {lbl}: var={np.var(g, ddof=1):.1f}, SD={np.std(g, ddof=1):.1f}")

print("\n" + "=" * 70)
print("3. PAIRED DATA COMPLETENESS")
print("=" * 70)
for tc, ids in [("single", range(1, 22)), ("multiple", range(22, 38))]:
    phone_ids = set(unified[(unified["target_count"] == tc) & (unified["device"] == "phone")]["participant_id"])
    lab_ids = set(unified[(unified["target_count"] == tc) & (unified["device"] == "lab")]["participant_id"])
    both = phone_ids & lab_ids
    phone_only = phone_ids - lab_ids
    lab_only = lab_ids - phone_ids
    print(f"  {tc}: {len(both)} complete pairs, {len(phone_only)} phone-only, {len(lab_only)} lab-only")
    if phone_only: print(f"    Phone only: {phone_only}")
    if lab_only: print(f"    Lab only: {lab_only}")

print("\n" + "=" * 70)
print("4. PHONE ACCURACY DISTRIBUTIONS")
print("=" * 70)
phone_agg = pd.concat([agg_phone_single, agg_phone_multiple], ignore_index=True)
for tc in ["single", "multiple"]:
    subset = phone_agg[phone_agg["target_count"] == tc]
    for var in ["mean_success_rate", "mean_hit_rate", "mean_false_alarms"]:
        vals = subset[var].dropna()
        print(f"  {tc}/{var} (n={len(vals)}):")
        print(f"    Mean={vals.mean():.2f}, Median={vals.median():.2f}, SD={vals.std():.2f}")
        print(f"    Min={vals.min():.2f}, Max={vals.max():.2f}")
        if len(vals) >= 3:
            stat, p = stats.shapiro(vals)
            print(f"    Shapiro-Wilk: W={stat:.4f}, p={p:.4f}")
        print()

print("=" * 70)
print("5. LAB TRIAL-LEVEL RT DISTRIBUTION")
print("=" * 70)
lab_all = pd.concat([lab_single, lab_multiple], ignore_index=True)
rt_vals = lab_all["response_time_ms"].dropna()
print(f"  Total trials: {len(rt_vals)}")
print(f"  Mean={rt_vals.mean():.1f}, Median={rt_vals.median():.1f}, SD={rt_vals.std():.1f}")
print(f"  Skewness: {stats.skew(rt_vals):.3f}")
print(f"  Kurtosis: {stats.kurtosis(rt_vals):.3f}")
stat, p = stats.shapiro(rt_vals[:5000] if len(rt_vals) > 5000 else rt_vals)
print(f"  Shapiro-Wilk: W={stat:.4f}, p={p:.4f}")

print("\n" + "=" * 70)
print("6. EFFECT SIZES")
print("=" * 70)

for tc in ["single", "multiple"]:
    phone_rt = unified[(unified["target_count"] == tc) & (unified["device"] == "phone")].set_index("participant_id")["mean_rt_ms"]
    lab_rt = unified[(unified["target_count"] == tc) & (unified["device"] == "lab")].set_index("participant_id")["mean_rt_ms"]
    common = phone_rt.index.intersection(lab_rt.index)
    diff = phone_rt.loc[common] - lab_rt.loc[common]
    d = diff.mean() / diff.std()
    print(f"  {tc} condition — Phone vs Lab (paired, n={len(common)}):")
    print(f"    Mean diff = {diff.mean():.1f} ms, SD of diff = {diff.std():.1f} ms")
    print(f"    Cohen's dz = {d:.3f}")
    t_stat, p_val = stats.ttest_rel(phone_rt.loc[common], lab_rt.loc[common])
    print(f"    Paired t-test: t={t_stat:.3f}, p={p_val:.6f}")
    print()

for dev in ["phone", "lab"]:
    single_rt = unified[(unified["target_count"] == "single") & (unified["device"] == dev)]["mean_rt_ms"]
    multiple_rt = unified[(unified["target_count"] == "multiple") & (unified["device"] == dev)]["mean_rt_ms"]
    pooled_sd = np.sqrt(((len(single_rt)-1)*single_rt.std()**2 + (len(multiple_rt)-1)*multiple_rt.std()**2) / (len(single_rt)+len(multiple_rt)-2))
    d = (single_rt.mean() - multiple_rt.mean()) / pooled_sd
    print(f"  {dev} — Single vs Multiple (independent):")
    print(f"    Single: M={single_rt.mean():.1f}, Multiple: M={multiple_rt.mean():.1f}")
    print(f"    Cohen's d = {d:.3f}")
    t_stat, p_val = stats.ttest_ind(single_rt, multiple_rt)
    print(f"    Independent t-test: t={t_stat:.3f}, p={p_val:.6f}")
    print()

print("=" * 70)
print("7. UNIFIED SUMMARY DATA (all 74 rows)")
print("=" * 70)
pd.set_option('display.max_rows', 80)
pd.set_option('display.width', 120)
print(unified.to_string(index=False))
