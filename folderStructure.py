from pathlib import Path

list_of_files = [

    # =========================
    # ROOT LEVEL
    # =========================
    ".gitignore",
    ".env",
    "requirements.txt",
    "README.md",

    # =========================
    # DATA DIRECTORY
    # =========================
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/train.jsonl",
    "data/val.jsonl",
    "data/test.jsonl",

    # =========================
    # NOTEBOOKS
    # =========================
    "notebooks/01_data_prep.ipynb",
    "notebooks/02_sft_training.ipynb",
    "notebooks/03_dpo_training.ipynb",
    "notebooks/04_evaluation.ipynb",

    # =========================
    # SCRIPTS
    # =========================
    "scripts/prepare_data.py",
    "scripts/evaluate.py",
    "scripts/generate_dpo_pairs.py",

    # =========================
    # MODELS
    # =========================
    "models/sft_lora/.gitkeep",
    "models/dpo_lora/.gitkeep",

    # =========================
    # RESULTS
    # =========================
    "results/sft_metrics.json",
    "results/dpo_metrics.json",
    "results/training_curves/.gitkeep"
]


# Create directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists():
        filepath.touch()
        print(f"Created: {filepath}")
    else:
        print(f"Already exists: {filepath}")