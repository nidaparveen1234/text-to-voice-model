from transformers import AutoProcessor, BarkModel
import torch
import scipy.io.wavfile
import gradio as gr

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small")
def text_to_speech(text):
    inputs = processor(text, voice_preset="v2/en_speaker_6")
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()
    scipy.io.wavfile.write("output.wav", model.generation_config.sample_rate, audio_array)
    return "output.wav"
app = gr.Interface(
    fn=text_to_speech,
    inputs=gr.Textbox(label="Enter Text to Convert"),
    outputs=gr.Audio(label="Generated Voice", type="filepath"),
    title="Text-to-Voice Generator (Bark Model)",
    description="Enter text and hear it spoken in an AI-generated voice."
)
app.launch()