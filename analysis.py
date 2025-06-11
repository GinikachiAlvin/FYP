import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
# import torch
# import eng_to_ipa as ipa
import uuid
import requests
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from fpdf import FPDF

np.complex = complex


API_URL = "https://thebickersteth-voxpreference.hf.space"

def transcribe_and_get_ipa_api(audio_path, word):
    with open(audio_path, "rb") as f:
        files = {"audioFile": f}
        data = {"word": word}  # Include the word to analyze
        response = requests.post(API_URL, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(result)
        text = result.get("text", "")
        ipa = result.get("ipa", "")
        return text, ipa
    else:
        raise ValueError(f"API error {response.status_code}: {response.text}")



# device = "cuda" if torch.cuda.is_available() else "cpu"
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)


def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# def transcribe(audio_path):
#     speech, sr = librosa.load(audio_path, sr=16000)
#     input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(device)
#     with torch.no_grad():
#         logits = model(input_values).logits
#     predicted_ids = torch.argmax(logits, dim=-1)
#     text = processor.batch_decode(predicted_ids)[0]
#     return text

# def get_ipa(text):
#     return ipa.convert(text.lower())


def format_confusion_matrix(confusion_data):
    headers = set()
    for row in confusion_data:
        headers.update(row.keys())
    headers = sorted(h for h in headers if h != 'ipa')
    table = ["IPA Variant | " + " | ".join(headers)]
    table.append("-" * len(table[0]))
    for row in confusion_data:
        line = row['ipa'] + " | " + " | ".join(str(row.get(h, 0)) for h in headers)
        table.append(line)
    return "\n".join(table)

def create_pdf_report(df, confusion, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Pronunciation Analysis Report", ln=True, align='C')

    pdf.ln(10)
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(200, 10, txt="IPA Variants and Clusters", ln=True)

    for i, row in df.iterrows():
        ipa = row.get("ipa", "")
        cluster = row.get("cluster", "")
        pdf.cell(200, 8, txt=f"{ipa} -> Cluster {cluster}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Confusion Matrix", ln=True)
    for entry in confusion:
        pdf.cell(200, 8, txt=str(entry), ln=True)

    pdf.output(filename)

def analyze_dataset(target_word, folder):
    data = []
    target_word_lower = target_word.lower()

    for fname in os.listdir(folder):
        if fname.endswith(".wav"):
            path = os.path.join(folder, fname)
            try:
                mfcc = extract_mfcc(path)
                text, ipa_full = transcribe_and_get_ipa_api(path, target_word_lower)

                print(text, ipa_full)
                if target_word_lower not in text.lower():
                    continue  # skip if word not found

                # Find matching IPA for the target word
                ipa_variants = []
                text_words = text.lower().split()
                ipa_words = ipa_full.split()

                for i, word in enumerate(text_words):
                    if target_word_lower in word and i < len(ipa_words):
                        ipa_variants.append(ipa_words[i])

                ipa = ipa_variants[0] if ipa_variants else ""


                data.append({
                    "filename": fname,
                    "text": text,
                    "ipa": ipa,
                    "word": target_word,
                    "features": mfcc
                })
            except Exception as e:
                print(f"Error processing {fname}: {e}")


    if not data:
        print(f"\n❌ No audio files contain the word '{target_word}'. Cannot proceed.")
        return {"error": f"No audio files contain the word '{target_word}'."}

    df = pd.DataFrame(data)
    features = np.stack(df["features"].values)
    features_scaled = StandardScaler().fit_transform(features)
    n_components = min(10, features_scaled.shape[0], features_scaled.shape[1])
    features_reduced = PCA(n_components=n_components).fit_transform(features_scaled)


    num_samples = len(df)
    if num_samples > 1:
        n_clusters = min(3, num_samples - 1)
        df["kmeans_cluster"] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(features_reduced)
        
        if n_clusters > 1:
            n_neighbors = min(10, num_samples - 1)
            df["spectral_cluster"] = SpectralClustering(
                n_clusters=n_clusters,
                affinity='nearest_neighbors',
                n_neighbors=n_neighbors
            ).fit_predict(features_reduced)
            df["hierarchical_cluster"] = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(features_reduced)
        else:
            df["spectral_cluster"] = 0
            df["hierarchical_cluster"] = 0
    else:
        df["kmeans_cluster"] = 0
        df["spectral_cluster"] = 0
        df["hierarchical_cluster"] = 0


    ipa_counts = df["ipa"].value_counts(normalize=True).to_dict()

    confusion = []
    ipa_clusters = df.groupby("ipa")["kmeans_cluster"].apply(list).to_dict()
    for ipa, clusters in ipa_clusters.items():
        counts = {f"cluster_{c}": clusters.count(c) for c in set(clusters)}
        row = {"ipa": ipa, **counts}
        confusion.append(row)

    print(f"\n✅ Target word: {target_word}\n")
    print("IPA variants:")
    for ipa in ipa_counts:
        print(f"  {ipa}")

    print("\nFrequency count:")
    for ipa, freq in ipa_counts.items():
        print(f"  {ipa} - {freq:.2f}")

    print("\nConfusion matrix:")
    print(format_confusion_matrix(confusion))

    pdf_path = os.path.join("backend", "results", f"report_{target_word.lower()}_{uuid.uuid4().hex}.pdf")
    create_pdf_report(df, confusion, pdf_path)

    print(f"\n📄 PDF report saved to: {pdf_path}")

    df["features"] = df["features"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return {
        "records": df.to_dict(orient="records"),
        "confusion": confusion,
        "pdf_path": pdf_path
    }
    


def export_analysis_to_csv(df, output_path="analysis_results.csv"):
    df.to_csv(output_path, index=False)
    return output_path

def analyze_pronunciations(file_paths, target_word):
    print("Analyzing files:", file_paths)
    print("Target word:", target_word)

    temp_folder = "backend/uploads_temp"
    os.makedirs(temp_folder, exist_ok=True)

    for path in file_paths:
        fname = os.path.basename(path)
        new_path = os.path.join(temp_folder, fname)
        os.rename(path, new_path)

    result = analyze_dataset(target_word, temp_folder)

    if "error" in result:
        raise ValueError(result["error"])

    return result["records"], result["confusion"], []

if __name__ == "__main__":
    print("=== Pronunciation Variant Analyzer ===")
    word = input("Enter the target word to analyze: ").strip()
    folder = input("Enter the path to the folder containing .wav files: ").strip()

    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist.")
    elif not word:
        print("Error: Target word cannot be empty.")
    else:
        analyze_dataset(word, folder)

