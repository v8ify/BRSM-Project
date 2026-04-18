import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings

# Importing existing cleaning/loading logic
from descriptive_statistics import (
    load_phone_data, load_lab_data, clean_phone_data,
    SINGLE_IDS, MULTIPLE_IDS
)

warnings.filterwarnings("ignore", category=FutureWarning)

# --- NEW: Logging Configuration ---
class Logger(object):
    """Captures terminal output and writes it to a file simultaneously."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# Define the results folder and filename
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_FILE = os.path.join(RESULTS_DIR, "per_level_statistical_results.txt")

# Start logging
sys.stdout = Logger(LOG_FILE)

# --- Existing Analysis Functions ---
def section(title):
    print(f"\n{'=' * 75}\n  {title}\n{'=' * 75}")

def subsection(title):
    print(f"\n  --- {title} ---\n")

def sig_marker(p, alpha=0.05):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < alpha: return "*"
    return "ns"

def build_level_data():
    """Loads and strictly deduplicates data for successful trials only[cite: 11]."""
    phone_s = clean_phone_data(load_phone_data("single", SINGLE_IDS))
    phone_m = clean_phone_data(load_phone_data("multiple", MULTIPLE_IDS))
    lab_s = load_lab_data("single", SINGLE_IDS)
    lab_m = load_lab_data("multiple", MULTIPLE_IDS)

    frames = []

    # Process Phone: FIRST successful completion only
    for df in [phone_s, phone_m]:
        subset = df[(df["Completed"] == True) & (df["Level"].isin(range(1, 11)))].copy()
        subset = subset.sort_values(["participant_id", "Level", "Timestamp"])
        subset = subset.drop_duplicates(subset=["participant_id", "Level"], keep="first")
        subset = subset.rename(columns={"InitialResponseTime(ms)": "rt_ms", "Level": "level"})
        frames.append(subset[["participant_id", "target_count", "device", "level", "rt_ms"]])

    # Process Lab: Map trials (0-9 -> 1-10)
    for df in [lab_s, lab_m]:
        subset = df[df["trials.thisN"].isin(range(10))].copy()
        subset["level"] = subset["trials.thisN"].astype(int) + 1
        subset = subset.drop_duplicates(subset=["participant_id", "level"], keep="first")
        subset = subset.rename(columns={"response_time_ms": "rt_ms"})
        frames.append(subset[["participant_id", "target_count", "device", "level", "rt_ms"]])

    return pd.concat(frames, ignore_index=True)

def run_analysis():
    df = build_level_data()
    all_results = []
    
    section("PER-LEVEL INFERENTIAL ANALYSIS (LEVELS 1-10)")

    for lvl in range(1, 11):
        lvl_df = df[df["level"] == lvl]
        if lvl_df.empty: continue
            
        subsection(f"LEVEL {lvl} RESULTS")

        # 1. Descriptives [cite: 11]
        desc = lvl_df.groupby(["target_count", "device"])["rt_ms"].agg(["mean", "std", "count"]).round(1)
        print(f"Descriptive Statistics - Level {lvl}:\n", desc)

        # 2. Device Effect (RQ3: Wilcoxon) [cite: 9, 14]
        for tc in ["single", "multiple"]:
            sub = lvl_df[lvl_df["target_count"] == tc]
            pivot = sub.pivot(index="participant_id", columns="device", values="rt_ms").dropna()
            
            if len(pivot) > 1:
                w_stat, p = stats.wilcoxon(pivot["phone"], pivot["lab"])
                all_results.append({
                    "Level": lvl, "Analysis": "Modality", "Group": tc, 
                    "Test": "Wilcoxon", "Stat": w_stat, "p": f"{p:.6f}", "Sig": sig_marker(p)
                })
                print(f"  Modality ({tc:<8}): W={w_stat:>6.1f}, p={p:.6f} {sig_marker(p)}")

        # 3. Target Load Effect (RQ2: Mann-Whitney) [cite: 8, 15]
        for dev in ["phone", "lab"]:
            s_vals = lvl_df[(lvl_df["device"] == dev) & (lvl_df["target_count"] == "single")]["rt_ms"]
            m_vals = lvl_df[(lvl_df["device"] == dev) & (lvl_df["target_count"] == "multiple")]["rt_ms"]
            
            if len(s_vals) > 0 and len(m_vals) > 0:
                u_stat, p = stats.mannwhitneyu(s_vals, m_vals)
                all_results.append({
                    "Level": lvl, "Analysis": "Load", "Group": dev, 
                    "Test": "Mann-Whitney", "Stat": u_stat, "p": f"{p:.6f}", "Sig": sig_marker(p)
                })
                print(f"  Load ({dev:<8}): U={u_stat:>6.1f}, p={p:.6f} {sig_marker(p)}")

    section("FINAL RESULTS SUMMARY TABLE")
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {LOG_FILE}")

if __name__ == "__main__":
    run_analysis()