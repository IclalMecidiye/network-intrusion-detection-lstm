"""
CICIDS2017 Veri Seti Indirme Araci
====================================
CICIDS2017 veri setini otomatik olarak indirir.

Kullanim:
    python src/download_dataset.py
    python src/download_dataset.py --output data/
"""

import os
import sys
import zipfile
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_DIR

# CIC-IDS2017 veri seti indirme URL'leri
# Not: Asil veri seti https://www.unb.ca/cic/datasets/ids-2017.html adresinden
# temin edilebilir. Kaggle uzerinden de indirilebilir.
DATASET_INFO = """
CICIDS2017 Veri Seti Indirme Talimatlari
==========================================

Veri setini asagidaki yontemlerden biri ile indirebilirsiniz:

1. KAGGLE UZERINDEN (Onerilen):
   - https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
   - veya: https://www.kaggle.com/datasets/hcavsi/cicids2017-dataset
   - Kaggle'dan indirip data/ klasorune cikartin

2. RESMI KAYNAK:
   - https://www.unb.ca/cic/datasets/ids-2017.html
   - "MachineLearningCSV.zip" dosyasini indirin
   - data/ klasorune cikartin

3. KAGGLE CLI ILE:
   pip install kaggle
   kaggle datasets download -d chethuhn/network-intrusion-dataset -p data/
   cd data && unzip network-intrusion-dataset.zip

CSV dosyalari data/ dizininde olmalidir:
  data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
  data/Friday-WorkingHours-Morning.pcap_ISCX.csv
  data/Monday-WorkingHours.pcap_ISCX.csv
  data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
  data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  data/Tuesday-WorkingHours.pcap_ISCX.csv
  data/Wednesday-workingHours.pcap_ISCX.csv
"""


def try_kaggle_download(output_dir):
    """Kaggle API ile veri setini indirmeyi dener."""
    try:
        import kaggle
        print("[*] Kaggle API ile indirme deneniyor...")
        kaggle.api.dataset_download_files(
            "chethuhn/network-intrusion-dataset",
            path=output_dir,
            unzip=True
        )
        print(f"[+] Veri seti basariyla indirildi: {output_dir}")
        return True
    except ImportError:
        print("[!] Kaggle kutuphanesi yuklu degil: pip install kaggle")
        return False
    except Exception as e:
        print(f"[!] Kaggle indirme hatasi: {e}")
        return False


def generate_sample_data(output_dir, n_samples=50000):
    """
    Test amacli sentetik CICIDS2017 benzeri veri seti olusturur.
    Gercek veri seti indirilemediginde kullanilir.

    Parameters
    ----------
    output_dir : str
        Cikti dizini.
    n_samples : int
        Toplam ornek sayisi.
    """
    import numpy as np
    import pandas as pd

    print(f"\n[*] Sentetik test verisi olusturuluyor ({n_samples} ornek)...")

    np.random.seed(42)

    # CICIDS2017 veri setindeki ozellik isimleri
    feature_names = [
        "Destination Port", "Flow Duration", "Total Fwd Packets",
        "Total Backward Packets", "Total Length of Fwd Packets",
        "Total Length of Bwd Packets", "Fwd Packet Length Max",
        "Fwd Packet Length Min", "Fwd Packet Length Mean",
        "Fwd Packet Length Std", "Bwd Packet Length Max",
        "Bwd Packet Length Min", "Bwd Packet Length Mean",
        "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
        "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
        "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
        "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
        "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
        "Min Packet Length", "Max Packet Length", "Packet Length Mean",
        "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
        "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
        "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
        "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
        "Avg Fwd Segment Size", "Avg Bwd Segment Size",
        "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
        "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
        "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
        "Subflow Fwd Bytes", "Subflow Bwd Packets",
        "Subflow Bwd Bytes", "Init_Win_bytes_forward",
        "Init_Win_bytes_backward", "act_data_pkt_fwd",
        "min_seg_size_forward", "Active Mean", "Active Std",
        "Active Max", "Active Min", "Idle Mean", "Idle Std",
        "Idle Max", "Idle Min",
    ]

    # Saldiri turleri ve oranlari
    labels_and_ratios = {
        "BENIGN": 0.80,
        "DoS Hulk": 0.05,
        "PortScan": 0.04,
        "DDoS": 0.04,
        "DoS GoldenEye": 0.02,
        "FTP-Patator": 0.015,
        "SSH-Patator": 0.01,
        "DoS slowloris": 0.005,
        "DoS Slowhttptest": 0.005,
        "Bot": 0.005,
        "Web Attack \u2013 Brute Force": 0.003,
        "Web Attack \u2013 XSS": 0.002,
        "Infiltration": 0.002,
        "Web Attack \u2013 Sql Injection": 0.001,
        "Heartbleed": 0.002,
    }

    data = {}
    for feature in feature_names:
        if "Flag" in feature or "Count" in feature:
            data[feature] = np.random.randint(0, 5, size=n_samples)
        elif "Port" in feature:
            data[feature] = np.random.randint(0, 65535, size=n_samples)
        elif "Ratio" in feature:
            data[feature] = np.random.uniform(0, 10, size=n_samples)
        elif "Bytes/s" in feature or "Packets/s" in feature:
            data[feature] = np.abs(np.random.exponential(1000, size=n_samples))
        else:
            data[feature] = np.abs(np.random.normal(100, 50, size=n_samples))

    # Etiketleri olustur
    labels = []
    for label, ratio in labels_and_ratios.items():
        count = int(n_samples * ratio)
        labels.extend([label] * count)
    # Kalan ornekleri BENIGN olarak doldur
    while len(labels) < n_samples:
        labels.append("BENIGN")
    labels = labels[:n_samples]
    np.random.shuffle(labels)
    data["Label"] = labels

    # Saldiri orneklerinin ozellik degerlerini farkli yap
    df = pd.DataFrame(data)
    attack_mask = df["Label"] != "BENIGN"

    # Saldiri orneklerinde bazi ozellikleri artir
    for col in ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s",
                "Fwd Packet Length Max", "Bwd Packet Length Max"]:
        if col in df.columns:
            df.loc[attack_mask, col] = df.loc[attack_mask, col] * np.random.uniform(
                2, 10, size=attack_mask.sum()
            )

    # CSV olarak kaydet
    csv_path = os.path.join(output_dir, "CICIDS2017_sample.csv")
    df.to_csv(csv_path, index=False)
    print(f"[+] Sentetik veri seti olusturuldu: {csv_path}")
    print(f"    Toplam ornek: {len(df)}")
    print(f"    Ozellik sayisi: {len(feature_names)}")
    print(f"    Saldiri ornekleri: {attack_mask.sum()} ({attack_mask.sum() / len(df) * 100:.1f}%)")
    print(f"    Normal ornekler: {(~attack_mask).sum()} ({(~attack_mask).sum() / len(df) * 100:.1f}%)")

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="CICIDS2017 Veri Seti Indirme Araci"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Cikti dizini (varsayilan: data/)"
    )
    parser.add_argument(
        "--generate-sample", action="store_true",
        help="Sentetik test verisi olustur"
    )
    parser.add_argument(
        "--n-samples", type=int, default=50000,
        help="Sentetik veri orneklem boyutu (varsayilan: 50000)"
    )

    args = parser.parse_args()

    output_dir = args.output or DATA_DIR
    os.makedirs(output_dir, exist_ok=True)

    if args.generate_sample:
        generate_sample_data(output_dir, args.n_samples)
        return

    # Veri seti zaten var mi kontrol et
    csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
    if csv_files:
        print(f"[+] {len(csv_files)} CSV dosyasi zaten mevcut: {output_dir}")
        for f in csv_files:
            filepath = os.path.join(output_dir, f)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"    - {f} ({size_mb:.1f} MB)")
        return

    # Kaggle ile dene
    success = try_kaggle_download(output_dir)

    if not success:
        print(DATASET_INFO)
        print("\n[*] Sentetik test verisi olusturuluyor...")
        generate_sample_data(output_dir, args.n_samples)


if __name__ == "__main__":
    main()
