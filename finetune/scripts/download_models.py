"""Pre-download all three base models so training can start immediately."""
import os
from huggingface_hub import snapshot_download

os.environ.setdefault("HF_HOME", "/mnt/data_lab/ChemQA/hf_cache")

MODELS = [
    "Qwen/Qwen2.5-14B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    # google/gemma-3-12b-it is gated without approval for this account; use the
    # unsloth mirror which ships identical weights.
    "unsloth/gemma-3-12b-it",
]

for m in MODELS:
    print(f"Downloading {m} ...", flush=True)
    path = snapshot_download(
        repo_id=m,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer.*", "*.txt"],
        max_workers=8,
    )
    print(f"  -> {path}", flush=True)

print("All downloads complete.")
