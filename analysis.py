import os
import uuid
import numpy as np
import pandas as pd
import librosa
import requests
from fpdf import FPDF
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

API_URL = "https://thebickersteth-voxpreference.hf.space"
FONT_PATH = "C:\\Users\\lenovo\\pronunciation_project\\DejaVuSans.ttf"

# Workaround for librosa deprecated np.complex
np.complex = complex


def transcribe_and_get_ipa_api(audio_path):
    with open(audio_path, "rb") as f:
        response = requests.post(API_URL, files={"audioFile": f})
    if response.status_code == 200:
        result = response.json()
        return result.get("transcription", ""), result.get("segments", [])
    raise ValueError(f"API error {response.status_code}: {response.text}")


def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    y, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


def format_confusion_matrix(confusion_data):
    headers = sorted({k for row in confusion_data for k in row if k != 'ipa'})
    lines = ["IPA Variant | " + " | ".join(headers), "-" * 50]
    for row in confusion_data:
        line = row['ipa'] + " | " + " | ".join(str(row.get(h, 0)) for h in headers)
        lines.append(line)
    return "\n".join(lines)


def create_pdf_report(df, confusion, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
    pdf.set_font("DejaVu", "", 12)

    pdf.cell(200, 10, "Pronunciation Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, "IPA Variants and Clusters:", ln=True)
    for _, row in df.iterrows():
        pdf.cell(200, 8, f"{row['ipa']} → Cluster {row['kmeans_cluster']}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, "Confusion Matrix:", ln=True)
    for line in format_confusion_matrix(confusion).splitlines():
        pdf.cell(200, 8, line.strip(), ln=True)

    pdf.output(filename)


def analyze_dataset(target_word, folder_path):
    target_word = target_word.lower()
    data = []

    for fname in os.listdir(folder_path):
        if not fname.endswith(".wav"):
            continue
        path = os.path.join(folder_path, fname)
        try:
            mfcc = extract_mfcc(path)
            transcription, segments = transcribe_and_get_ipa_api(path)

            if target_word not in transcription.lower():
                continue

            ipa = next((seg["ipa"] for seg in segments if seg["text"].lower() == target_word and seg.get("ipa")), "")
            if not ipa:
                continue

            data.append({
                "filename": fname,
                "text": transcription,
                "ipa": ipa,
                "word": target_word,
                "features": mfcc
            })

        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")

    if not data:
        error_msg = f"No audio files contain the word '{target_word}'."
        print(f"\n❌ {error_msg}")
        return {"error": error_msg}

    df = pd.DataFrame(data)

    # Clustering
    features = np.stack(df["features"].values)
    scaled = StandardScaler().fit_transform(features)
    n_components = min(10, *scaled.shape)
    reduced = PCA(n_components=n_components).fit_transform(scaled)

    num_samples = len(df)
    n_clusters = min(3, num_samples - 1) if num_samples > 1 else 1

    df["kmeans_cluster"] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(reduced)
    df["spectral_cluster"] = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors').fit_predict(reduced) if n_clusters > 1 else 0
    df["hierarchical_cluster"] = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(reduced) if n_clusters > 1 else 0

    # Confusion matrix from cluster-IPA relationship
    confusion = []
    for ipa, group in df.groupby("ipa"):
        counts = {f"cluster_{i}": group["kmeans_cluster"].tolist().count(i) for i in set(group["kmeans_cluster"])}
        confusion.append({"ipa": ipa, **counts})

    # Output results
    ipa_counts = df["ipa"].value_counts(normalize=True).to_dict()

    print(f"\n✅ Target word: {target_word}")
    print("IPA variants:")
    for ipa in ipa_counts: print(f"  {ipa}")
    print("\nFrequency count:")
    for ipa, freq in ipa_counts.items(): print(f"  {ipa} - {freq:.2f}")
    print("\nConfusion matrix:")
    print(format_confusion_matrix(confusion))

    # Save PDF report
    output_path = os.path.join("backend", "results", f"report_{target_word}_{uuid.uuid4().hex}.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_pdf_report(df, confusion, output_path)
    print(f"\n📄 PDF report saved to: {output_path}")

    df["features"] = df["features"].apply(lambda x: x.tolist())
    return {
        "records": df.to_dict(orient="records"),
        "confusion": confusion,
        "pdf_path": output_path
    }


def export_analysis_to_csv(df, output_path="analysis_results.csv"):
    df.to_csv(output_path, index=False)
    return output_path


def analyze_pronunciations(file_paths, target_word):
    print("Analyzing files:", file_paths)
    temp_folder = "backend/uploads_temp"
    os.makedirs(temp_folder, exist_ok=True)

    for path in file_paths:
        os.rename(path, os.path.join(temp_folder, os.path.basename(path)))

    result = analyze_dataset(target_word, temp_folder)
    if "error" in result:
        raise ValueError(result["error"])
    return result["records"], result["confusion"], []


if __name__ == "__main__":
    print("=== Pronunciation Variant Analyzer ===")
    word = input("Enter the target word to analyze: ").strip()
    folder = input("Enter the path to the folder containing .wav files: ").strip()

    if not os.path.exists(folder):
        print(f"❌ Error: Folder '{folder}' does not exist.")
    elif not word:
        print("❌ Error: Target word cannot be empty.")
    else:
        analyze_dataset(word, folder)
