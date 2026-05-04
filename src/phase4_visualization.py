"""
Phase 4: Visualization & Presentation Graphics
- Safety-Fairness tradeoff bar chart
- XAI bias mechanism bar chart
- Radar chart (bias fingerprint)
- Heatmap (mechanism intensity)
- Contrast chart (wealthy vs marginalized per model)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import GRADED_DATA_CSV, OUTPUT_DIR

sns.set_theme(style="whitegrid")


def _get_m_percentages(df, model_name, persona_name):
    subset = df[(df["Model"] == model_name) & (df["Persona"] == persona_name)]
    total = len(subset)
    if total == 0:
        return [0] * 5
    cols = ["M1_Anchoring", "M2_NonCompliance", "M3_DeferredCare", "M4_Substitution", "M5_ResponsibilityShift"]
    return [(subset[c] == True).sum() / total * 100 for c in cols]


def chart_safety_fairness(safety_scores, fairness_scores):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ["GPT-4o-mini (Cloud)", "BioMistral-7B (Local)"]
    x = np.arange(len(models))
    w = 0.35

    r1 = ax.bar(x - w / 2, safety_scores, w, label="Clinical Safety (Max 4)", color="#2b5c8f")
    r2 = ax.bar(x + w / 2, fairness_scores, w, label="Fairness Parity (Max 4)", color="#e07a5f")

    ax.set_ylabel("Score (Out of 4.0)", fontsize=12, fontweight="bold")
    ax.set_title("The Medical AI Tradeoff: Clinical Capability vs. Socioeconomic Fairness", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 4.5)

    for rect in list(r1) + list(r2):
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Tradeoff_Chart.png", dpi=300)
    plt.show()


def chart_xai_bars(gpt_m, bio_m):
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = ["M1: Anchoring", "M2: Non-Compliance", "M3: Deferred Care", "M4: Substitution", "M5: Responsibility Shift"]
    x = np.arange(len(labels))
    w = 0.35

    r1 = ax.bar(x - w / 2, gpt_m, w, label="GPT-4o-mini", color="#2b5c8f")
    r2 = ax.bar(x + w / 2, bio_m, w, label="BioMistral-7B", color="#81b29a")

    ax.set_ylabel("Trigger Frequency (%)", fontsize=12, fontweight="bold")
    ax.set_title("XAI Extraction: Mechanisms of Bias in Marginalized Personas", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)

    for rect in list(r1) + list(r2):
        h = rect.get_height()
        ax.annotate(f"{h}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "XAI_Mechanisms_Chart.png", dpi=300)
    plt.show()


def chart_radar(gpt_m, bio_m):
    labels = ["M1: Anchoring", "M2: Non-Compliance", "M3: Deferred Care", "M4: Substitution", "M5: Responsibility Shift"]
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], labels, size=10, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80], ["20%", "40%", "60%", "80%"], color="grey", size=8)
    plt.ylim(0, 100)

    for vals, label, color in [(gpt_m, "GPT-4o-mini", "#2b5c8f"), (bio_m, "BioMistral-7B", "#81b29a")]:
        v = vals + vals[:1]
        ax.plot(angles, v, linewidth=2, label=label, color=color)
        ax.fill(angles, v, color, alpha=0.1)

    plt.title("AI Bias Fingerprint (Radar Chart)", size=14, fontweight="bold", y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Radar_Chart.png", dpi=300)
    plt.show()


def chart_heatmap(gpt_m, bio_m):
    fig, ax = plt.subplots(figsize=(10, 4))
    cols = ["M1: Anchoring", "M2: Non-Compliance", "M3: Deferred Care", "M4: Substitution", "M5: Responsibility Shift"]
    data = pd.DataFrame([gpt_m, bio_m], columns=cols, index=["GPT-4o-mini", "BioMistral-7B"])
    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Trigger Frequency (%)"}, linewidths=0.5, ax=ax)
    plt.title("Heatmap: Concentration of Sociological Bias Mechanisms", fontsize=14, fontweight="bold", pad=20)
    plt.yticks(rotation=0, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Bias_Heatmap.png", dpi=300)
    plt.show()


def chart_contrast(graded_csv: str = None):
    df = pd.read_csv(graded_csv or str(GRADED_DATA_CSV))
    labels = ["M1\nAnchoring", "M2\nNon-Compliance", "M3\nDeferred Care", "M4\nSubstitution", "M5\nResponsibility Shift"]
    x = np.arange(len(labels))
    w = 0.35

    gpt_w = _get_m_percentages(df, "GPT-4o-mini", "Wealthy")
    gpt_l = _get_m_percentages(df, "GPT-4o-mini", "Low_Income")
    bio_w = _get_m_percentages(df, "BioMistral-7B", "Wealthy")
    bio_l = _get_m_percentages(df, "BioMistral-7B", "Low_Income")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    def _plot(ax, wealthy, marg, title):
        r1 = ax.bar(x - w / 2, wealthy, w, label="Wealthy Executive", color="#2b5c8f")
        r2 = ax.bar(x + w / 2, marg, w, label="Low-Income / Uninsured", color="#e07a5f")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend()
        for rect in list(r1) + list(r2):
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    _plot(ax1, gpt_w, gpt_l, "GPT-4o-mini: Sociological Prejudice")
    ax1.set_ylabel("Trigger Frequency (%)", fontsize=12, fontweight="bold")
    _plot(ax2, bio_w, bio_l, "BioMistral-7B: Egalitarian Incompetence")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Contrast_XAI_Chart.png", dpi=300)
    plt.show()


def run_visualizations(safety_scores=None, fairness_scores=None, gpt_m=None, bio_m=None):
    # Use provided scores or defaults from the notebook results
    safety = safety_scores or [1.78, 0.56]
    fairness = fairness_scores or [3.44, 3.60]
    gpt = gpt_m or [32.0, 24.0, 40.0, 2.0, 38.7]
    bio = bio_m or [14.0, 14.0, 59.3, 0.0, 86.0]

    chart_safety_fairness(safety, fairness)
    chart_xai_bars(gpt, bio)
    chart_radar(gpt, bio)
    chart_heatmap(gpt, bio)
    chart_contrast()
    print(f"\nAll charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_visualizations()
