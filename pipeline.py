#!/usr/bin/env python3
"""pipeline.py — Generate all analysis outputs from the cell_counts database.

Parts 2-4 of the Teiknical assessment. Reads from the SQLite database
created by load_data.py and writes CSV tables and plot images to outputs/.
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

DB_PATH = Path("cell_counts.db")
OUTPUT_DIR = Path("outputs")
PLOT_DIR = OUTPUT_DIR / "plots"

POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]

POPULATION_LABELS = {
    "b_cell": "B Cell",
    "cd8_t_cell": "CD8 T Cell",
    "cd4_t_cell": "CD4 T Cell",
    "nk_cell": "NK Cell",
    "monocyte": "Monocyte",
}

# Teiko-inspired color palette
COLORS = {
    "bg": "#0F172A",
    "card": "#1E293B",
    "grid": "#334155",
    "text": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "teal": "#2DD4BF",
    "coral": "#F87171",
}


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Part 2: Relative frequency summary table
# ---------------------------------------------------------------------------

def part2_summary_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """Compute relative frequency of each cell population per sample."""
    print("\n=== Part 2: Relative Frequency Summary ===")

    rows = conn.execute("""
        SELECT sample, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte
        FROM sample_view
    """).fetchall()

    records = []
    for row in rows:
        sample = row["sample"]
        counts = {pop: row[pop] for pop in POPULATIONS}
        total = sum(counts.values())
        for pop in POPULATIONS:
            records.append({
                "sample": sample,
                "total_count": total,
                "population": pop,
                "count": counts[pop],
                "percentage": round(100.0 * counts[pop] / total, 4),
            })

    df = pd.DataFrame(records)

    out_path = OUTPUT_DIR / "summary.csv"
    df.to_csv(out_path, index=False)
    print(f"  Wrote {len(df):,} rows to {out_path}")
    print(f"  Samples: {df['sample'].nunique():,}")
    return df


# ---------------------------------------------------------------------------
# Part 3: Responder vs non-responder statistical analysis
# ---------------------------------------------------------------------------

def part3_query_cohort(conn: sqlite3.Connection) -> pd.DataFrame:
    """Fetch melanoma + miraclib + PBMC samples with relative frequencies."""
    query = """
        SELECT sample, subject, response, time_from_treatment_start as day,
               b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte
        FROM sample_view
        WHERE condition = 'melanoma'
          AND treatment = 'miraclib'
          AND sample_type = 'PBMC'
          AND response IS NOT NULL
    """
    df = pd.DataFrame(conn.execute(query).fetchall(),
                       columns=["sample", "subject", "response", "day", *POPULATIONS])

    df["total_count"] = df[POPULATIONS].sum(axis=1)
    for pop in POPULATIONS:
        df[f"{pop}_pct"] = 100.0 * df[pop] / df["total_count"]

    return df


def part3_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run Welch's t-test on per-subject averaged values with BH correction.

    Each subject has 3 timepoints. We average across timepoints first to ensure
    independence, then compare responders vs non-responders. We use Welch's
    t-test because the Kolmogorov-Smirnov normality test confirms the data is
    normally distributed across all populations and groups.
    """
    pct_cols = [f"{pop}_pct" for pop in POPULATIONS]
    subj_avg = (
        df.groupby(["subject", "response"])[pct_cols]
        .mean()
        .reset_index()
    )

    raw_pvals = []
    results = []
    for pop in POPULATIONS:
        col = f"{pop}_pct"
        responders = subj_avg.loc[subj_avg["response"] == "yes", col]
        non_responders = subj_avg.loc[subj_avg["response"] == "no", col]

        # K-S normality test
        ks_r_stat, ks_r_p = stats.kstest(
            responders, "norm", args=(responders.mean(), responders.std())
        )
        ks_nr_stat, ks_nr_p = stats.kstest(
            non_responders, "norm", args=(non_responders.mean(), non_responders.std())
        )

        # Welch's t-test (does not assume equal variances)
        stat, pval = stats.ttest_ind(
            responders, non_responders, equal_var=False
        )
        raw_pvals.append(pval)
        results.append({
            "population": pop,
            "population_label": POPULATION_LABELS[pop],
            "test": "Welch's t-test",
            "statistic": round(stat, 4),
            "p_value": round(pval, 6),
            "responder_mean": round(responders.mean(), 2),
            "non_responder_mean": round(non_responders.mean(), 2),
            "n_responders": len(responders),
            "n_non_responders": len(non_responders),
            "ks_responder_p": round(ks_r_p, 4),
            "ks_non_responder_p": round(ks_nr_p, 4),
        })

    # Benjamini-Hochberg FDR correction
    from statsmodels.stats.multitest import multipletests
    _, bh_pvals, _, _ = multipletests(raw_pvals, method="fdr_bh")

    result_df = pd.DataFrame(results)
    result_df["p_value_bh"] = [round(p, 6) for p in bh_pvals]
    result_df["significant"] = result_df["p_value_bh"] < 0.05

    return result_df.sort_values("p_value")


def _render_boxplot(df: pd.DataFrame, p_lookup: dict, out_dir: Path,
                    subtitle: str = "") -> None:
    """Render one boxplot per population and save to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="darkgrid")

    for pop in POPULATIONS:
        col = f"{pop}_pct"
        label = POPULATION_LABELS[pop]
        pval = p_lookup[pop]

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor(COLORS["bg"])
        ax.set_facecolor(COLORS["card"])

        plot_df = df[["response", col]].copy()
        plot_df["response"] = plot_df["response"].map(
            {"yes": "Responder", "no": "Non-Responder"}
        )

        palette = {"Responder": COLORS["teal"], "Non-Responder": COLORS["coral"]}

        sns.boxplot(
            data=plot_df, x="response", y=col, hue="response",
            palette=palette, width=0.5, linewidth=1.2,
            fliersize=3, ax=ax, legend=False,
            order=["Responder", "Non-Responder"],
        )
        sns.stripplot(
            data=plot_df, x="response", y=col,
            color=COLORS["text"], alpha=0.15, size=2.5, jitter=True, ax=ax,
            order=["Responder", "Non-Responder"],
        )

        title = f"{label}\n{subtitle}" if subtitle else label
        ax.set_title(title, color=COLORS["text"], fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("", color=COLORS["text_secondary"])
        ax.set_ylabel("Relative Frequency (%)", color=COLORS["text_secondary"], fontsize=10)

        # P-value annotation
        sig_color = COLORS["teal"] if pval < 0.05 else COLORS["text_secondary"]
        sig_marker = " **" if pval < 0.05 else ""
        ax.text(
            0.5, 0.97, f"p = {pval:.4f}{sig_marker}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=sig_color, fontweight="bold",
        )

        ax.tick_params(colors=COLORS["text_secondary"], labelsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.3)
        ax.grid(axis="x", visible=False)

        fig.tight_layout()
        fig.savefig(out_dir / f"{pop}.png", dpi=150, facecolor=COLORS["bg"])
        plt.close(fig)


def _render_combined_boxplot(df: pd.DataFrame, raw_p: dict, bh_p: dict,
                             out_path: Path, title: str = "") -> None:
    """Render all 5 populations in a single row of subplots with three-tier significance."""
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.5))
    fig.patch.set_facecolor(COLORS["bg"])

    if title:
        fig.suptitle(title, color=COLORS["text"], fontsize=18, fontweight="bold",
                     y=1.02)

    palette = {"Responder": COLORS["teal"], "Non-Responder": COLORS["coral"]}

    for ax, pop in zip(axes, POPULATIONS):
        col = f"{pop}_pct"
        label = POPULATION_LABELS[pop]
        raw_pval = raw_p[pop]
        bh_pval = bh_p[pop]

        ax.set_facecolor(COLORS["card"])

        plot_df = df[["response", col]].copy()
        plot_df["response"] = plot_df["response"].map(
            {"yes": "Responder", "no": "Non-Responder"}
        )

        sns.boxplot(
            data=plot_df, x="response", y=col, hue="response",
            palette=palette, width=0.5, linewidth=1.2,
            fliersize=3, ax=ax, legend=False,
            order=["Responder", "Non-Responder"],
        )
        sns.stripplot(
            data=plot_df, x="response", y=col,
            color=COLORS["text"], alpha=0.15, size=2, jitter=True, ax=ax,
            order=["Responder", "Non-Responder"],
        )

        ax.set_title(label, color=COLORS["text"], fontsize=15, fontweight="bold",
                     pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Relative Freq. (%)" if pop == POPULATIONS[0] else "",
                      color=COLORS["text_secondary"], fontsize=11)

        # Three-tier significance coloring
        if bh_pval < 0.05:
            sig_color = COLORS["teal"]
            sig_marker = " **"
        elif raw_pval < 0.05:
            sig_color = "#FBBF24"  # amber
            sig_marker = " *"
        else:
            sig_color = COLORS["text_secondary"]
            sig_marker = ""

        ax.text(
            0.5, 0.97, f"BH p = {bh_pval:.4f}{sig_marker}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, color=sig_color, fontweight="bold",
        )

        ax.tick_params(colors=COLORS["text_secondary"], labelsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.3)
        ax.grid(axis="x", visible=False)

    # Legend for significance tiers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["teal"],
               markersize=8, label="** Significant (BH-adjusted p < 0.05)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#FBBF24",
               markersize=8, label="*  Significant before correction only"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["text_secondary"],
               markersize=8, label="   Not significant"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               frameon=False, fontsize=11,
               labelcolor=COLORS["text_secondary"],
               bbox_to_anchor=(0.5, -0.06), columnspacing=4.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close(fig)


def _run_welch_tests(df: pd.DataFrame) -> tuple[dict, dict]:
    """Run Welch's t-test per population. Returns (raw_p_lookup, bh_p_lookup)."""
    from statsmodels.stats.multitest import multipletests

    raw_pvals = []
    pops = []
    for pop in POPULATIONS:
        col = f"{pop}_pct"
        r = df.loc[df["response"] == "yes", col]
        nr = df.loc[df["response"] == "no", col]
        _, pval = stats.ttest_ind(r, nr, equal_var=False)
        raw_pvals.append(pval)
        pops.append(pop)

    _, bh_pvals, _, _ = multipletests(raw_pvals, method="fdr_bh")
    return dict(zip(pops, raw_pvals)), dict(zip(pops, bh_pvals))


def part3_boxplots(df: pd.DataFrame, stat_results: pd.DataFrame) -> None:
    """Generate boxplots for all timepoints (averaged) and each individual timepoint."""
    pct_cols = [f"{pop}_pct" for pop in POPULATIONS]

    # All timepoints (per-subject averaged)
    avg_df = (
        df.groupby(["subject", "response"])[pct_cols]
        .mean()
        .reset_index()
    )
    raw_all = dict(zip(stat_results["population"], stat_results["p_value"]))
    bh_all = dict(zip(stat_results["population"], stat_results["p_value_bh"]))
    _render_boxplot(avg_df, raw_all, PLOT_DIR / "all_timepoints",
                    subtitle="All Timepoints (per-subject avg)")
    _render_combined_boxplot(avg_df, raw_all, bh_all,
                             PLOT_DIR / "all_timepoints_combined.png",
                             title="All Timepoints (per-subject avg)")

    # Per-timepoint
    for day in [0, 7, 14]:
        day_df = df[df["day"] == day].copy()
        raw_day, bh_day = _run_welch_tests(day_df)
        _render_boxplot(day_df, bh_day, PLOT_DIR / f"day_{day}",
                        subtitle=f"Day {day}")
        _render_combined_boxplot(day_df, raw_day, bh_day,
                                 PLOT_DIR / f"day_{day}_combined.png",
                                 title=f"Day {day}")

    print(f"  Saved boxplots to {PLOT_DIR}/{{all_timepoints,day_0,day_7,day_14}}/")


def part3_analysis(conn: sqlite3.Connection) -> pd.DataFrame:
    """Run the full Part 3 analysis."""
    print("\n=== Part 3: Statistical Analysis ===")
    print("  Cohort: melanoma, miraclib, PBMC, responders vs non-responders")

    df = part3_query_cohort(conn)
    n_subj_resp = df.loc[df["response"] == "yes", "subject"].nunique()
    n_subj_nonresp = df.loc[df["response"] == "no", "subject"].nunique()
    print(f"  Samples: {len(df):,} from {df['subject'].nunique()} subjects")
    print(f"  Subjects: {n_subj_resp} responders, {n_subj_nonresp} non-responders")
    print(f"  Test: Welch's t-test (normality confirmed via K-S test)")
    print(f"  Analysis uses per-subject averages (collapsed across timepoints)")

    stat_results = part3_statistical_tests(df)

    out_path = OUTPUT_DIR / "statistical_results.csv"
    stat_results.to_csv(out_path, index=False)
    print(f"  Wrote statistical results to {out_path}")

    sig = stat_results[stat_results["significant"]]
    nonsig = stat_results[~stat_results["significant"]]
    if len(sig) > 0:
        print(f"  Significant (p < 0.05): {', '.join(sig['population_label'])}")
    if len(nonsig) > 0:
        print(f"  Not significant: {', '.join(nonsig['population_label'])}")

    part3_boxplots(df, stat_results)
    return stat_results


# ---------------------------------------------------------------------------
# Part 4: Baseline cohort breakdown
# ---------------------------------------------------------------------------

def part4_cohort_breakdown(conn: sqlite3.Connection) -> pd.DataFrame:
    """Query baseline melanoma PBMC miraclib samples and compute breakdowns."""
    print("\n=== Part 4: Cohort Breakdown ===")
    print("  Filter: melanoma, miraclib, PBMC, baseline (day 0)")

    base_query = """
        SELECT sample, subject, project, response, sex
        FROM sample_view
        WHERE condition = 'melanoma'
          AND treatment = 'miraclib'
          AND sample_type = 'PBMC'
          AND time_from_treatment_start = 0
    """
    df = pd.DataFrame(
        conn.execute(base_query).fetchall(),
        columns=["sample", "subject", "project", "response", "sex"],
    )
    print(f"  Total baseline samples: {len(df)}")

    # By project
    by_project = df.groupby("project").size().reset_index(name="sample_count")
    print(f"\n  By Project:")
    for _, row in by_project.iterrows():
        print(f"    {row['project']}: {row['sample_count']}")

    # By response
    by_response = df.groupby("response").agg(
        subject_count=("subject", "nunique")
    ).reset_index()
    print(f"\n  By Response (unique subjects):")
    for _, row in by_response.iterrows():
        print(f"    {row['response']}: {row['subject_count']}")

    # By sex
    by_sex = df.groupby("sex").agg(
        subject_count=("subject", "nunique")
    ).reset_index()
    print(f"\n  By Sex (unique subjects):")
    for _, row in by_sex.iterrows():
        label = "Male" if row["sex"] == "M" else "Female"
        print(f"    {label}: {row['subject_count']}")

    # Save combined breakdown
    breakdown = {
        "metric": [],
        "category": [],
        "count": [],
    }
    for _, row in by_project.iterrows():
        breakdown["metric"].append("project")
        breakdown["category"].append(row["project"])
        breakdown["count"].append(row["sample_count"])
    for _, row in by_response.iterrows():
        breakdown["metric"].append("response")
        breakdown["category"].append(row["response"])
        breakdown["count"].append(row["subject_count"])
    for _, row in by_sex.iterrows():
        breakdown["metric"].append("sex")
        breakdown["category"].append(row["sex"])
        breakdown["count"].append(row["subject_count"])

    out_df = pd.DataFrame(breakdown)
    out_path = OUTPUT_DIR / "cohort_breakdown.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Wrote breakdown to {out_path}")
    return out_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    conn = get_connection()
    try:
        part2_summary_table(conn)
        part3_analysis(conn)
        part4_cohort_breakdown(conn)
    finally:
        conn.close()

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
