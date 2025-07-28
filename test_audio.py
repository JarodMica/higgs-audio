from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

import torch
import torchaudio
import time
import click
import yaml
import os

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
with open("personal/tokens.yaml", "r") as f:
    tokens = yaml.safe_load(f)
    
HF_TOKEN = tokens["hf_token"]
# Set the HF token as environment variable for huggingface_hub authentication
os.environ["HF_TOKEN"] = HF_TOKEN

system_prompt = (
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)

messages = [
    Message(
        role="system",
        content=system_prompt,
    ),
    Message(
        role="user",
        content="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
    ),
]
device = "cuda" if torch.cuda.is_available() else "cpu"

serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

while True:
    input("Press enter to generate audio:")
    start_time = time.time()
    
    output: HiggsAudioResponse = serve_engine.generate(
        chat_ml_sample=ChatMLSample(messages=messages),
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.95,
        top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
    torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")