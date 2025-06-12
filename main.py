import gradio as gr
import tempfile
import os
from analyse import analyze_dataset

def run_analysis(audio_files, target_word):
    if not audio_files or not target_word:
        return "Please upload audio files and enter a target word."

    with tempfile.TemporaryDirectory() as temp_dir:
        for audio in audio_files:
            filepath = os.path.join(temp_dir, audio.name)
            with open(filepath, "wb") as f:
                f.write(audio.read())

        result = analyze_dataset(target_word, temp_dir)
        if "error" in result:
            return result["error"]
        
        return f"✅ Analyzed '{target_word}'\n\nIPA Variants:\n" + \
               "\n".join([rec['ipa'] for rec in result['records']])

iface = gr.Interface(
    fn=run_analysis,
    inputs=[
        gr.File(file_types=[".wav"], label="Upload WAV files", multiple=True),
        gr.Textbox(label="Target Word")
    ],
    outputs="text",
    title="Pronunciation Variant Analyzer",
    description="Upload .wav files and enter a target word to analyze IPA variants and cluster pronunciations.",
)

if __name__ == "__main__":
    iface.launch()
