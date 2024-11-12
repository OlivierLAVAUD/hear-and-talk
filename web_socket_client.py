# [Hear and Talk] FastAPI
# L'interface radio avec Whisper et gTTS en architecture Websocket (partie client)

import gradio as gr
import asyncio
import json
import websockets

async def transcribe_audio(audio, language):
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        data = {
            "type": "transcribe",
            "audio": audio,
            "language": language,
        }
        await websocket.send(json.dumps(data))
        response = await websocket.recv()
        return json.loads(response)["text"]

async def synthesize_text(text, language):
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        data = {
            "type": "synthesize",
            "text": text,
            "language": language,
        }
        await websocket.send(json.dumps(data))
        response = await websocket.recv()
        return json.loads(response)["audio"]

with gr.Blocks() as demo:
    gr.Markdown("## Transcription vocale et synthèse vocale avec Whisper et gTTS")

    with gr.Row():
        gr.Markdown("### Langues disponibles")
        language_choice = gr.Dropdown(choices=["en", "fr", "de", "es", "it", "pt", "nl"], value="en", label="Langue")

    audio_input = gr.Audio(type="filepath", label="Enregistrer un audio")

    text_output = gr.Textbox(label="Texte transcrit", interactive=False)
    transcription_button = gr.Button("Transcrire l'audio")
    transcription_button.click(fn=transcribe_audio, inputs=[audio_input, language_choice], outputs=text_output)

    text_input = gr.Textbox(label="Entrer du texte pour la synthèse vocale")
    audio_output = gr.Audio(label="Résultat audio")
    synthesize_button = gr.Button("Synthétiser le texte en audio")
    synthesize_button.click(fn=synthesize_text, inputs=[text_input, language_choice], outputs=audio_output)

demo.launch()
