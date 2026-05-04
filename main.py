"""
The Persona Gap: Auditing In-Context Bias in General vs. Specialized Medical LLMs
==================================================================================
Main orchestrator — run all phases sequentially or individually.

Usage:
    python main.py              # Run all phases
    python main.py phase1       # Only data engineering
    python main.py phase2       # Only inference
    python main.py phase3       # Only evaluation/grading
    python main.py phase4       # Only visualization
"""

import sys

from src.phase1_data_engineering import run_phase1
from src.phase2_inference import run_phase2
from src.phase3_evaluation import run_grading, calculate_aggregate_scores, calculate_bias_mechanisms
from src.phase4_visualization import run_visualizations


def main():
    phases = sys.argv[1:] if len(sys.argv) > 1 else ["phase1", "phase2", "phase3", "phase4"]

    if "phase1" in phases:
        print("\n" + "=" * 60)
        print("PHASE 1: Data Engineering & Persona Methodology")
        print("=" * 60)
        run_phase1()

    if "phase2" in phases:
        print("\n" + "=" * 60)
        print("PHASE 2: Generative Inference Engine")
        print("=" * 60)
        run_phase2()

    if "phase3" in phases:
        print("\n" + "=" * 60)
        print("PHASE 3: AI Judge Evaluation")
        print("=" * 60)
        run_grading()
        calculate_aggregate_scores()
        calculate_bias_mechanisms()

    if "phase4" in phases:
        print("\n" + "=" * 60)
        print("PHASE 4: Visualization")
        print("=" * 60)
        run_visualizations()

    print("\nDone.")


if __name__ == "__main__":
    main()
