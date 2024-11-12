# [Hear and Talk] Straight
# L'interface radio avec Whisper et gTTS


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

def change_model(model_choice):
    global pipe
    pipe = initialize_whisper_model(model_choice)

with gr.Blocks() as demo:
    gr.Markdown("## Hear and Talk (Whisper, gTTS, FastAPI")

    with gr.Row():
        gr.Markdown("### Choisir un modèle Whisper")
        model_choice = gr.Dropdown(choices=["openai/whisper-small", "openai/whisper-medium", "openai/whisper-large-v3"],
                                   value="openai/whisper-small", label="Modèle Whisper")
        model_choice.change(change_model, inputs=model_choice, outputs=[])

    with gr.Row():
        gr.Markdown("### Langues disponibles")
        language_choice = gr.Dropdown(choices=["en", "fr", "de", "es", "it", "pt", "nl"], value="en", label="Langue")

    audio_input = gr.Audio(type="filepath", label="Enregistrer un audio")

    text_output = gr.Textbox(label="Texte transcrit", interactive=False)
    transcription_button = gr.Button("Transcrire l'audio")
    transcription_button.click(speech_to_text, inputs=[audio_input, language_choice], outputs=text_output)

    text_input = gr.Textbox(label="Entrer du texte pour la synthèse vocale")
    audio_output = gr.Audio(label="Résultat audio")
    synthesize_button = gr.Button("Synthétiser le texte en audio")
    synthesize_button.click(text_to_speech, inputs=[text_input, language_choice], outputs=audio_output)

demo.launch()
