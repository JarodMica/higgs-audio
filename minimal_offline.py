"""Minimal TTS inference example with fully offline models."""

import os
import torch
import torchaudio
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

# Disable HuggingFace hub connectivity checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Local model paths using models/ folder (run download_models.py first)
MODEL_PATH = "models/higgs-audio-model"
AUDIO_TOKENIZER_PATH = "models/higgs-audio-tokenizer"

# Reference audio settings (optional - set to None to use smart voice selection)
AUDIO_REF_PATH =  "voice_prompts/belinda.wav"  # Path to reference audio file
AUDIO_REF_TRANSCRIPT_PATH =  "voice_prompts/belinda.txt"  # Path to reference transcript

SYSTEM_PROMPT = "Generate audio while speaking very slowly and with a lot of pauses."

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_messages_with_reference_audio(text_to_generate, audio_ref_path=None, audio_ref_transcript_path=None):
    """Create ChatML messages with optional reference audio for voice cloning."""
    
    global SYSTEM_PROMPT
    
    messages = [Message(role="system", content=SYSTEM_PROMPT)]
    
    if audio_ref_path and audio_ref_transcript_path:
        if not os.path.exists(audio_ref_path):
            raise FileNotFoundError(f"Reference audio file not found: {audio_ref_path}")
        if not os.path.exists(audio_ref_transcript_path):
            raise FileNotFoundError(f"Reference transcript file not found: {audio_ref_transcript_path}")
            
        with open(audio_ref_transcript_path, "r", encoding="utf-8") as f:
            ref_transcript = f.read().strip()
        
        messages.extend([
            Message(role="user", content=ref_transcript),
            Message(role="assistant", content=AudioContent(audio_url=audio_ref_path)),
        ])
    
    messages.append(Message(role="user", content=text_to_generate))
    
    return messages

try:
    serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)
    
    text_to_generate = "Why is EVERYONE and their grandmother shopping right now? It's like a zombie apocalypse in here."
    
    while True:
        cont_audio = input("Continue? (y/n)")
    
        if cont_audio == "n":
            break
        
        change_system_prompt = input("Change system prompt? (y/n)")
        
        if change_system_prompt == "y":
            SYSTEM_PROMPT = input("Enter new system prompt: ")
        
        text_to_generate = input("Enter text to generate: ")
        
        messages = create_messages_with_reference_audio(
            text_to_generate=text_to_generate,
            audio_ref_path=AUDIO_REF_PATH,
            audio_ref_transcript_path=AUDIO_REF_TRANSCRIPT_PATH
        )

        output = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )

        torchaudio.save("output_offline.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        print("✓ Successfully generated audio offline!")
        if AUDIO_REF_PATH:
            print(f"✓ Used voice cloning with reference: {AUDIO_REF_PATH}")
        else:
            print("✓ Used smart voice selection (no reference audio)")

except Exception as e:
    print(f"Error: {e}")
    print("Make sure to update MODEL_PATH and AUDIO_TOKENIZER_PATH with correct local paths")