"""
Download ONNX MiniLM model for sentence embeddings.
Run once before ranking (pre-computation step).
Model: Xenova/all-MiniLM-L6-v2 from HuggingFace Hub.
"""
import json
import os
from pathlib import Path

BASE = Path(__file__).parent
MODEL_DIR = BASE / "models" / "minilm-onnx"

FILES = [
    "onnx/model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
]

def download():
    print(f"Downloading Xenova/all-MiniLM-L6-v2 to {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface-hub...")
        import subprocess
        subprocess.run(["python", "-m", "pip", "install", "huggingface-hub", "-q"], check=True)
        from huggingface_hub import hf_hub_download

    repo = "Xenova/all-MiniLM-L6-v2"

    for fpath in FILES:
        local = MODEL_DIR / os.path.basename(fpath)
        if local.exists():
            print(f"  Already cached: {local.name}")
            continue
        print(f"  Downloading {fpath}...")
        hf_hub_download(repo_id=repo, filename=fpath, local_dir=MODEL_DIR, local_dir_use_symlinks=False)

    print(f"\nModel cached at {MODEL_DIR}")
    print("Files:")
    for f in sorted(MODEL_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    download()
