# The Persona Gap: Auditing In-Context Bias in General vs. Specialized Medical LLMs

A High-Risk Generative AI Architecture that audits whether LLMs offer different clinical advice based on socioeconomic or gender personas.

## Key Findings: The Safety-Fairness Paradox
This audit revealed a critical tradeoff in medical AI:
* **Cloud Generalists (GPT-4o-mini):** Demonstrated high clinical utility (1.78/2.0) but suffered from significant socioeconomic anchoring.
* **Clinical Specialists (BioMistral-7B):** Achieved statistical fairness, but did so through **"Clinical Paralysis"**—shifting responsibility and refusing to triage in 86% of marginalized cases.

## Data and Paper Availability
* **Full Paper:** [Link to arXiv pending]
* **Reproducibility Archive:** The full synthetic dataset, inference outputs, and execution snapshot are permanently archived on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)



## Project Structure

```
HRP/
├── main.py                         # Orchestrator — run all or individual phases
├── config.py                       # Shared settings, paths, personas, seed data
├── requirements.txt                # Python dependencies
├── .env.example                    # Template for API keys
├── src/
│   ├── phase1_data_engineering.py  # Synthetic data gen (TVAE), vignettes, persona injection
│   ├── phase2_inference.py         # GPT-4o-mini + BioMistral-7B inference
│   ├── phase3_evaluation.py        # AI Judge grading, fairness & bias metrics
│   └── phase4_visualization.py     # All charts (bar, radar, heatmap, contrast)
└── output/                         # Generated CSVs and chart PNGs
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

```bash
# Run everything end-to-end
python main.py

# Run individual phases
python main.py phase1    # Generate synthetic data + prompts
python main.py phase2    # Run model inference (needs GPU for BioMistral)
python main.py phase3    # Grade responses with AI Judge
python main.py phase4    # Generate all visualizations
```

## Notes

- **BioMistral-7B** requires a CUDA GPU with ~8GB VRAM (4-bit quantized). On CPU-only machines it will be skipped with a placeholder.
- **GPT-4o-mini** and **GPT-4o** (judge) require a valid OpenAI API key.
- All outputs (CSVs, PNGs) are saved to the `output/` directory.


## Citation
If you use this code or dataset in your research, please cite the accompanying paper:
```bibtex
@article{chaudhari2026personagap,
  title={The Persona Gap: Auditing In-Context Sociological Bias in General vs. Specialized LLMs for Clinical Triage},
  author={Chaudhari, Manish},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}