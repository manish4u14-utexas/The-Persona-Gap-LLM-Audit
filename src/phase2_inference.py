"""
Phase 2: Generative Inference Engine
- Query GPT-4o-mini (cloud generalist)
- Query BioMistral-7B (local specialist, requires GPU)
- Run hybrid inference across all 250 prompts
"""

import os
import time
import pandas as pd
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    OPENAI_API_KEY, GPT_MODEL, BIOMISTRAL_MODEL_ID,
    PROMPTS_CSV, GENERATED_ADVICE_CSV,
)

# ── OpenAI (Cloud Generalist) ──────────────────────────────────────────────

_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def query_generalist(prompt_text: str) -> str:
    try:
        resp = _get_openai_client().chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert triage physician. Provide concise clinical assessments."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI Error: {e}"


# ── BioMistral (Local Specialist) ──────────────────────────────────────────

_specialist_model = None
_specialist_tokenizer = None


def _load_specialist():
    global _specialist_model, _specialist_tokenizer
    if _specialist_model is not None:
        return

    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. BioMistral requires a GPU with bitsandbytes 4-bit quantization.")
        print("BioMistral queries will return placeholder text on CPU-only machines.")
        return

    print("Downloading and loading BioMistral-7B (4-bit quantized)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    _specialist_tokenizer = AutoTokenizer.from_pretrained(BIOMISTRAL_MODEL_ID)
    _specialist_model = AutoModelForCausalLM.from_pretrained(
        BIOMISTRAL_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
    )
    print("BioMistral loaded into GPU memory.")


def query_specialist(prompt_text: str) -> str:
    _load_specialist()
    if _specialist_model is None:
        return "[BioMistral skipped — no GPU available]"

    formatted = f"<s>[INST] You are a triage physician. Evaluate this patient:\n{prompt_text} [/INST]"
    inputs = _specialist_tokenizer(formatted, return_tensors="pt").to("cuda")
    outputs = _specialist_model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=True)
    full_text = _specialist_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.split("[/INST]")[-1].strip()


# ── Hybrid Inference Pipeline ──────────────────────────────────────────────

def run_phase2(prompts_csv: str = None):
    prompts_csv = prompts_csv or str(PROMPTS_CSV)
    master_df = pd.read_csv(prompts_csv)
    print(f"Starting hybrid inference for {len(master_df)} prompts...")

    for index, row in master_df.iterrows():
        gpt_advice = query_generalist(row["Full_Prompt"])
        bio_advice = query_specialist(row["Full_Prompt"])

        generated_row = pd.DataFrame([{
            "Case_ID": row["Case_ID"],
            "Persona": row["Persona"],
            "Chief_Complaint": row["Chief_Complaint"],
            "Full_Prompt": row["Full_Prompt"],
            "GPT4o_Mini_Advice": gpt_advice,
            "BioMistral_Advice": bio_advice,
        }])

        header = not os.path.isfile(GENERATED_ADVICE_CSV)
        generated_row.to_csv(GENERATED_ADVICE_CSV, mode="a", header=header, index=False)

        if (index + 1) % 10 == 0:
            print(f"  Processed {index + 1}/{len(master_df)}...")
        time.sleep(0.5)

    print(f"\nInference complete. Saved to {GENERATED_ADVICE_CSV}")


if __name__ == "__main__":
    run_phase2()
