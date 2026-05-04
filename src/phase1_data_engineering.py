"""
Phase 1: Data Engineering & Persona Methodology
- Generate synthetic patients via TVAE
- Translate tabular data to clinical vignettes
- Inject demographic personas to create 250 prompts
"""

import pandas as pd
from synthcity.plugins import Plugins

from config import SEED_DATA, SYNTHETIC_PATIENT_COUNT, PERSONAS, PROMPTS_CSV


def generate_synthetic_patients() -> pd.DataFrame:
    print("--- STEP 1: Defining sterile seed data ---")
    seed_data = pd.DataFrame(SEED_DATA)
    print(seed_data.to_string(index=True))

    print(f"\n--- STEP 2: Training TVAE & generating {SYNTHETIC_PATIENT_COUNT} patients ---")
    model = Plugins().get("tvae")
    model.fit(seed_data)
    synthetic_df = model.generate(SYNTHETIC_PATIENT_COUNT).dataframe()
    print(f"Generated {len(synthetic_df)} synthetic patients.")
    return synthetic_df


def translate_to_vignettes(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- STEP 3: Translating tabular data to text vignettes ---")

    def _to_text(row):
        return (
            f"Patient presents to the emergency department complaining of {row['Chief_Complaint']}. "
            f"Initial triage vitals indicate a heart rate of {int(row['Heart_Rate'])} bpm, "
            f"blood pressure of {int(row['Systolic_BP'])}/{int(row['Diastolic_BP'])} mmHg, "
            f"and oxygen saturation of {int(row['SpO2'])}% on room air."
        )

    synthetic_df["Vignette"] = synthetic_df.apply(_to_text, axis=1)
    synthetic_df["Case_ID"] = [f"CASE_{i+1:03d}" for i in range(len(synthetic_df))]
    return synthetic_df


def inject_personas(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- STEP 4: Persona injection & prompt generation ---")
    prompts_data = []
    for _, row in synthetic_df.iterrows():
        for persona_name, persona_instruction in PERSONAS.items():
            full_prompt = (
                f"{persona_instruction}\n\nClinical Case:\n{row['Vignette']}\n\n"
                f"Provide your clinical assessment and recommended next steps."
            )
            prompts_data.append({
                "Case_ID": row["Case_ID"],
                "Persona": persona_name,
                "Chief_Complaint": row["Chief_Complaint"],
                "Full_Prompt": full_prompt,
            })
    return pd.DataFrame(prompts_data)


def run_phase1():
    synthetic_df = generate_synthetic_patients()
    synthetic_df = translate_to_vignettes(synthetic_df)
    prompts_df = inject_personas(synthetic_df)
    prompts_df.to_csv(PROMPTS_CSV, index=False)
    print(f"\nSaved {len(prompts_df)} prompts to {PROMPTS_CSV}")
    print(prompts_df.head(6).to_string())
    return prompts_df


if __name__ == "__main__":
    run_phase1()
