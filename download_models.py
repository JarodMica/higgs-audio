"""Download all required models to local models/ folder for offline usage."""

import os
from huggingface_hub import snapshot_download
from pathlib import Path
import yaml

# Create models directory
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print("Downloading models for offline usage...")

with open("personal/tokens.yaml", "r") as f:
    tokens = yaml.safe_load(f)
    
HF_TOKEN = tokens["hf_token"]
# Set the HF token as environment variable for huggingface_hub authentication
os.environ["HF_TOKEN"] = HF_TOKEN

# Main models required by HiggsAudioServeEngine
models_to_download = [
    # Main model and tokenizer
    ("bosonai/higgs-audio-v2-generation-3B-base", "higgs-audio-model"),
    ("bosonai/higgs-audio-v2-tokenizer", "higgs-audio-tokenizer"),
    
    # Semantic models used by the audio tokenizer
    ("bosonai/hubert_base", "hubert_base"),
    ("facebook/hubert-base-ls960", "hubert-base-ls960"),
    ("microsoft/wavlm-base-plus", "wavlm-base-plus"),
    
    # Whisper model (used if encode_whisper_embed is enabled)
    ("openai/whisper-large-v3-turbo", "whisper-large-v3-turbo"),
]

for hf_model_name, local_name in models_to_download:
    local_path = models_dir / local_name
    
    if local_path.exists():
        print(f"✓ {local_name} already exists, skipping...")
        continue
    
    print(f"Downloading {hf_model_name} -> models/{local_name}")
    try:
        snapshot_download(
            repo_id=hf_model_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,  # Don't use symlinks for portability
        )
        print(f"✓ Successfully downloaded {local_name}")
    except Exception as e:
        print(f"✗ Failed to download {hf_model_name}: {e}")

print("\nDownload complete! Models are now available in the 'models/' folder.")
print("You can now use the models offline by pointing to these local paths.")