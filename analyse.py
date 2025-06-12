import os
import pandas as pd
import librosa
import torch
import requests
from io import StringIO
from fpdf import FPDF
from openai import OpenAI
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from tkinter import Tk, filedialog

# --- CONFIG ---
client = OpenAI(api_key="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")
API_URL = "https://thebickersteth-voxpreference.hf.space"
FONT_PATH = "C:\\Users\\lenovo\\pronunciation_project\\DejaVuSans.ttf"

# --- IPA VARIANT GENERATION ---
def get_ipa_variants(target_word, count):
    variant_count = min(count, 4)
    prompt = f"""
You're a phonetics expert. The word is "{target_word}".
The word came up {count} time(s) 
Generate {variant_count} different IPA English variant(s) that could be used for this word due to accent or pronunciation differences. I only need the variants, no description what so ever.
If the number of variants asked to be produce is 1, then the frequency count for the variant generated is (100%)
If the number of variants asked to be produce is 2, then the frequency count for the variants generated are (50%, 50%)%
If the number of variants asked to be produce is 3, then the frequency count for the variants generated are (33%, 33% & 34%)
If the number of variants asked to be produce is 4, then the frequency count for each variant generated should be distruibuted amongst the four variancs since the appeared {count} tume(s).
Return a table like this:
IPA Variant | Frequency | Fraction (between 0 and 1)
The frequency must be an integer, and the fractions must sum up to 1.0.
Then generate a confusion matrix comparing the 4 variants. Matrix values should also be between 0 and 1.
"""
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- SAVE OUTPUT ---
def save_outputs(word, output, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    lines = output.strip().splitlines()
    header = next((line for line in lines if "IPA" in line and "|" in line), None)

    if not header:
        raise ValueError("⚠️ Table header not found in output.")

    expected_cols = len(header.split("|")) - 2
    table_lines = [line for line in lines if "|" in line and line.count("|") == expected_cols + 1]
    confusion_lines = [line for line in lines if line.startswith("|") and line.count("|") >= 4]

    df = pd.read_csv(StringIO("\n".join(table_lines)), sep="|", engine="python", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    csv_path = os.path.join(output_dir, f"{word}_analysis.csv")
    pdf_path = os.path.join(output_dir, f"{word}_analysis.pdf")
    df.to_csv(csv_path, index=False)

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.cell(200, 10, f"IPA Analysis Report for '{word}'", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, "Confusion Matrix:", ln=True)
    for line in confusion_lines:
        pdf.cell(200, 10, line.strip(), ln=True)

    pdf.output(pdf_path)
    return csv_path, pdf_path

# --- API TRANSCRIPTION ---
def transcribe_and_get_ipa_api(audio_path):
    with open(audio_path, "rb") as f:
        response = requests.post(API_URL, files={"audioFile": f})
    if response.status_code == 200:
        result = response.json()
        return result.get("transcription", ""), result.get("segments", [])
    raise ValueError(f"API error {response.status_code}: {response.text}")

# --- MAIN WORKFLOW ---
def main():
    Tk().withdraw()
    print("📁 Select an audio file...")
    audio_path = filedialog.askopenfilename(title="Select an Audio File", filetypes=[("Audio Files", "*.wav *.mp3 *.m4a")])

    if not audio_path:
        print("❌ No file selected.")
        return

    target_word = input("Enter the target word: ").strip().lower()
    print(f"\n📡 Sending file to API: {API_URL}")
    transcription, segments = transcribe_and_get_ipa_api(audio_path)

    count = sum(1 for seg in segments if seg["text"].lower() == target_word)
    if count == 0:
        print(f"\n❌ The word '{target_word}' was not found in the transcription.")
        return

    print(f"\n✅ Found '{target_word}' {count} time(s). Generating IPA variants...")
    output = get_ipa_variants(target_word, count)

    print("\n===== RAW GPT OUTPUT =====\n", output, "\n===== END =====\n")
    print("💾 Saving results...")
    csv_path, pdf_path = save_outputs(target_word, output)
    print(f"\n🎉 Done!\nCSV: {csv_path}\nPDF: {pdf_path}")
    os.startfile(pdf_path)

# --- ENTRY ---
if __name__ == "__main__":
    main()
