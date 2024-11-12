# [Hear and Talk] FastAPI
# L'interface radio avec Whisper et gTTS en architecture Websocket (partie serveur) 

import asyncio
import websockets
import json
import gradio as gr
import torch
import librosa
from transformers import pipeline
from gtts import gTTS
import tempfile

def initialize_whisper_model(model_name="openai/whisper-small"):
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )

model_name = "openai/whisper-small"
pipe = initialize_whisper_model(model_name)
print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

def speech_to_text(audio, language="en"):
    try:
        audio_data, sample_rate = librosa.load(audio, sr=16000, mono=True)
        result = pipe(inputs=audio_data, return_timestamps=True, chunk_length_s=30)
        return result["text"]
    except Exception as e:
        return f"An error occurred: {str(e)}"

def text_to_speech(text, language="fr"):
    try:
        tts = gTTS(text=text, lang=language)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        return f"An error occurred: {str(e)}"

async def handle_connection(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        if data["type"] == "transcribe":
            audio = data["audio"]
            language = data["language"]
            text = speech_to_text(audio, language)
            response = {"type": "transcription_result", "text": text}
        elif data["type"] == "synthesize":
            text = data["text"]
            language = data["language"]
            audio = text_to_speech(text, language)
            response = {"type": "synthesis_result", "audio": audio}
        else:
            response = {"type": "error", "message": "Invalid request type"}

        await websocket.send(json.dumps(response))

start_server = websockets.serve(handle_connection, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
