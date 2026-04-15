"""
Veri On Isleme Modulu
=====================
CICIDS2017 veri setinin yuklenmesi, temizlenmesi ve
LSTM modeli icin hazirlanmasi islemlerini gerceklestirir.

Adimlar:
1. CSV dosyalarinin yuklenmesi ve birlestirilmesi
2. Eksik/hatalı verilerin temizlenmesi
3. Ozellik secimi ve normalizasyon
4. Etiketlerin kodlanmasi (Label Encoding)
5. Egitim/Test/Dogrulama setlerine ayirma
6. LSTM icin 3 boyutlu veri yapisina donusturme
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    DATA_DIR, CICIDS2017_FILES, TEST_SIZE, VALIDATION_SIZE,
    RANDOM_STATE, BINARY_CLASSIFICATION, ATTACK_LABELS, MULTICLASS_LABELS
)


def load_cicids2017(data_dir=None, sample_size=None):
    """
    CICIDS2017 veri setini yukler ve birlestirir.

    Parameters
    ----------
    data_dir : str, optional
        Veri dizini yolu. None ise config'deki DATA_DIR kullanilir.
    sample_size : int, optional
        Her dosyadan alinacak ornek sayisi. None ise tum veri kullanilir.

    Returns
    -------
    pd.DataFrame
        Birlestirilen veri seti.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    dataframes = []
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(
            f"'{data_dir}' dizininde CSV dosyasi bulunamadi.\n"
            f"Lutfen CICIDS2017 veri setini '{data_dir}' dizinine indirin.\n"
            f"Indirme linki: https://www.unb.ca/cic/datasets/ids-2017.html"
        )

    print(f"[*] {len(csv_files)} adet CSV dosyasi bulundu.")

    for csv_file in tqdm(csv_files, desc="Veri yukleniyor"):
        file_path = os.path.join(data_dir, csv_file)
        try:
            df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)

        # Sutun isimlerindeki bosluklari temizle
        df.columns = df.columns.str.strip()

        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=RANDOM_STATE)

        dataframes.append(df)
        print(f"  -> {csv_file}: {len(df)} satir, {len(df.columns)} sutun")

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\n[+] Toplam veri boyutu: {combined_df.shape[0]} satir, {combined_df.shape[1]} sutun")

    return combined_df


def clean_data(df):
    """
    Veri setindeki eksik, sonsuz ve hatali verileri temizler.

    Parameters
    ----------
    df : pd.DataFrame
        Ham veri seti.

    Returns
    -------
    pd.DataFrame
        Temizlenmis veri seti.
    """
    print("\n[*] Veri temizleme basladi...")
    initial_rows = len(df)

    # 1. Sutun isimlerini temizle
    df.columns = df.columns.str.strip()

    # 2. Gereksiz sutunlari kaldir (IP, Port, Flow ID gibi)
    columns_to_drop = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in
               ["flow id", "source ip", "destination ip",
                "source port", "destination port", "timestamp"]):
            columns_to_drop.append(col)

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors="ignore")
        print(f"  -> {len(columns_to_drop)} gereksiz sutun kaldirildi: {columns_to_drop}")

    # 3. Label sutununu bul
    label_col = None
    for col in df.columns:
        if col.strip().lower() == "label":
            label_col = col
            break

    if label_col is None:
        raise ValueError("'Label' sutunu bulunamadi!")

    # 4. Sayisal olmayan sutunlari kaldir (Label haric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in df.columns if c not in numeric_cols and c != label_col]
    if non_numeric:
        df = df.drop(columns=non_numeric, errors="ignore")
        print(f"  -> {len(non_numeric)} sayisal olmayan sutun kaldirildi: {non_numeric}")

    # 5. Sonsuz degerleri NaN'a cevir
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # 6. NaN degerleri kaldir
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        df = df.dropna()
        print(f"  -> {nan_count} adet NaN deger temizlendi")

    # 7. Duplicate satirlari kaldir
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"  -> {dup_count} adet tekrarlayan satir kaldirildi")

    final_rows = len(df)
    print(f"  -> Temizleme tamamlandi: {initial_rows} -> {final_rows} satir "
          f"({initial_rows - final_rows} satir cikarildi)")

    return df


def encode_labels(df, binary=None):
    """
    Etiketleri sayisal degerlere donusturur.

    Parameters
    ----------
    df : pd.DataFrame
        Temizlenmis veri seti.
    binary : bool, optional
        True ise ikili siniflandirma (Normal/Saldiri).
        None ise config'deki BINARY_CLASSIFICATION kullanilir.

    Returns
    -------
    tuple
        (X, y, label_encoder, label_names)
    """
    if binary is None:
        binary = BINARY_CLASSIFICATION

    # Label sutununu bul
    label_col = None
    for col in df.columns:
        if col.strip().lower() == "label":
            label_col = col
            break

    if label_col is None:
        raise ValueError("'Label' sutunu bulunamadi!")

    print(f"\n[*] Etiket kodlama basladi (mod: {'Ikili' if binary else 'Coklu'})...")

    # Etiket dagilimlari
    print("\n  Etiket dagilimi:")
    label_counts = df[label_col].value_counts()
    for label, count in label_counts.items():
        print(f"    {label}: {count} ({count / len(df) * 100:.2f}%)")

    if binary:
        # Ikili siniflandirma: BENIGN=0, Saldiri=1
        df["encoded_label"] = df[label_col].apply(
            lambda x: 0 if x.strip() == "BENIGN" else 1
        )
        label_names = ["Normal (BENIGN)", "Saldiri (Attack)"]
    else:
        # Coklu siniflandirma
        label_mapping = MULTICLASS_LABELS
        df["encoded_label"] = df[label_col].map(
            lambda x: label_mapping.get(x.strip(), -1)
        )
        # Eslesmeyen etiketleri kaldir
        df = df[df["encoded_label"] != -1]
        label_names = [name for _, name in sorted(
            {v: k for k, v in label_mapping.items()}.items()
        )]

    # Ozellik matrisi ve etiket vektoru
    X = df.drop(columns=[label_col, "encoded_label"]).values.astype(np.float32)
    y = df["encoded_label"].values.astype(np.int32)

    feature_names = [c for c in df.columns if c not in [label_col, "encoded_label"]]

    print(f"\n  -> Ozellik sayisi: {X.shape[1]}")
    print(f"  -> Toplam ornek: {X.shape[0]}")
    print(f"  -> Sinif sayisi: {len(np.unique(y))}")

    return X, y, feature_names, label_names


def normalize_features(X_train, X_test, X_val=None):
    """
    Ozellikleri StandardScaler ile normalize eder.

    Parameters
    ----------
    X_train : np.ndarray
        Egitim verisi.
    X_test : np.ndarray
        Test verisi.
    X_val : np.ndarray, optional
        Dogrulama verisi.

    Returns
    -------
    tuple
        Normalize edilmis veriler ve scaler nesnesi.
    """
    print("\n[*] Ozellik normalizasyonu (StandardScaler)...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_test_scaled, X_val_scaled, scaler

    return X_train_scaled, X_test_scaled, scaler


def reshape_for_lstm(X):
    """
    Veriyi LSTM icin 3 boyutlu (samples, timesteps, features) formata donusturur.
    Her ornegi tek bir zaman adimi olarak ele aliyoruz.

    Parameters
    ----------
    X : np.ndarray
        2 boyutlu ozellik matrisi (samples, features).

    Returns
    -------
    np.ndarray
        3 boyutlu LSTM girdisi (samples, 1, features).
    """
    return X.reshape((X.shape[0], 1, X.shape[1]))


def prepare_data(data_dir=None, sample_size=None, binary=None):
    """
    Tum veri on isleme adimlarini birlestiren ana fonksiyon.

    Parameters
    ----------
    data_dir : str, optional
        Veri dizini yolu.
    sample_size : int, optional
        Her dosyadan alinacak ornek sayisi.
    binary : bool, optional
        Ikili siniflandirma mi?

    Returns
    -------
    dict
        Islenmis veri setini iceren sozluk:
        - X_train, X_test, X_val: Egitim/Test/Dogrulama ozellikleri
        - X_train_lstm, X_test_lstm, X_val_lstm: LSTM icin reshape edilmis
        - y_train, y_test, y_val: Etiketler
        - scaler: StandardScaler nesnesi
        - feature_names: Ozellik isimleri
        - label_names: Etiket isimleri
        - n_features: Ozellik sayisi
        - n_classes: Sinif sayisi
    """
    print("=" * 60)
    print("  CICIDS2017 Veri On Isleme")
    print("=" * 60)

    # 1. Veri yukleme
    df = load_cicids2017(data_dir=data_dir, sample_size=sample_size)

    # 2. Veri temizleme
    df = clean_data(df)

    # 3. Etiket kodlama
    X, y, feature_names, label_names = encode_labels(df, binary=binary)

    # 4. Egitim/Test/Dogrulama setlerine ayirma
    print("\n[*] Veri bolme islemi...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"  -> Egitim seti: {X_train.shape[0]} ornek")
    print(f"  -> Test seti: {X_test.shape[0]} ornek")
    print(f"  -> Dogrulama seti: {X_val.shape[0]} ornek")

    # 5. Normalizasyon
    X_train_scaled, X_test_scaled, X_val_scaled, scaler = normalize_features(
        X_train, X_test, X_val
    )

    # 6. LSTM icin reshape
    print("\n[*] LSTM icin veri donusumu (3D)...")
    X_train_lstm = reshape_for_lstm(X_train_scaled)
    X_test_lstm = reshape_for_lstm(X_test_scaled)
    X_val_lstm = reshape_for_lstm(X_val_scaled)
    print(f"  -> LSTM girdi boyutu: {X_train_lstm.shape}")

    n_features = X_train_scaled.shape[1]
    n_classes = len(np.unique(y))

    print("\n" + "=" * 60)
    print("  Veri On Isleme Tamamlandi!")
    print("=" * 60)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_val": X_val_scaled,
        "X_train_lstm": X_train_lstm,
        "X_test_lstm": X_test_lstm,
        "X_val_lstm": X_val_lstm,
        "y_train": y_train,
        "y_test": y_test,
        "y_val": y_val,
        "scaler": scaler,
        "feature_names": feature_names,
        "label_names": label_names,
        "n_features": n_features,
        "n_classes": n_classes,
    }


if __name__ == "__main__":
    # Test icin calistirma
    data = prepare_data(sample_size=10000)
    print(f"\nOzellik sayisi: {data['n_features']}")
    print(f"Sinif sayisi: {data['n_classes']}")
    print(f"LSTM egitim seti boyutu: {data['X_train_lstm'].shape}")
