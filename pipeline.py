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
        SELECT sample, response, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte
        FROM sample_view
        WHERE condition = 'melanoma'
          AND treatment = 'miraclib'
          AND sample_type = 'PBMC'
          AND response IS NOT NULL
    """
    df = pd.DataFrame(conn.execute(query).fetchall(),
                       columns=["sample", "response", *POPULATIONS])

    df["total_count"] = df[POPULATIONS].sum(axis=1)
    for pop in POPULATIONS:
        df[f"{pop}_pct"] = 100.0 * df[pop] / df["total_count"]

    return df


def part3_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run Mann-Whitney U tests for each cell population."""
    results = []
    for pop in POPULATIONS:
        col = f"{pop}_pct"
        responders = df.loc[df["response"] == "yes", col]
        non_responders = df.loc[df["response"] == "no", col]

        stat, pval = stats.mannwhitneyu(
            responders, non_responders, alternative="two-sided"
        )
        results.append({
            "population": pop,
            "population_label": POPULATION_LABELS[pop],
            "test": "Mann-Whitney U",
            "statistic": round(stat, 2),
            "p_value": round(pval, 6),
            "significant": pval < 0.05,
            "responder_median": round(responders.median(), 2),
            "non_responder_median": round(non_responders.median(), 2),
        })

    return pd.DataFrame(results).sort_values("p_value")


def part3_boxplots(df: pd.DataFrame, stat_results: pd.DataFrame) -> None:
    """Generate one boxplot per cell population comparing responders vs non-responders."""
    sns.set_theme(style="darkgrid")

    p_lookup = dict(zip(stat_results["population"], stat_results["p_value"]))

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

        ax.set_title(label, color=COLORS["text"], fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("", color=COLORS["text_secondary"])
        ax.set_ylabel("Relative Frequency (%)", color=COLORS["text_secondary"], fontsize=10)

        # P-value annotation
        sig_color = COLORS["teal"] if pval < 0.05 else COLORS["text_secondary"]
        sig_marker = " *" if pval < 0.05 else ""
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
        fig.savefig(PLOT_DIR / f"{pop}.png", dpi=150, facecolor=COLORS["bg"])
        plt.close(fig)

    print(f"  Saved 5 boxplots to {PLOT_DIR}/")


def part3_analysis(conn: sqlite3.Connection) -> pd.DataFrame:
    """Run the full Part 3 analysis."""
    print("\n=== Part 3: Statistical Analysis ===")
    print("  Cohort: melanoma, miraclib, PBMC, responders vs non-responders")

    df = part3_query_cohort(conn)
    n_resp = (df["response"] == "yes").sum()
    n_nonresp = (df["response"] == "no").sum()
    print(f"  Samples: {len(df):,} (responders: {n_resp:,}, non-responders: {n_nonresp:,})")

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
