"""
Gercek Zamanli Tahmin Modulu
==============================
Egitilmis LSTM modelini kullanarak yeni ag trafigi
verilerinde saldiri tespiti yapar.

Bu modul, canli siber guvenlik loglari uzerinde
gercek zamanli saldiri tespiti icin tasarlanmistir.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODELS_DIR, DATA_DIR


def load_prediction_pipeline(model_type="lstm"):
    """
    Tahmin icin gerekli model ve scaler'i yukler.

    Parameters
    ----------
    model_type : str
        Model tipi: "lstm", "rf", "nb"

    Returns
    -------
    tuple
        (model, scaler)
    """
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler dosyasi bulunamadi: {scaler_path}\n"
            "Lutfen once train.py ile modeli egitin."
        )
    scaler = joblib.load(scaler_path)

    if model_type == "lstm":
        from tensorflow.keras.models import load_model
        model_path = os.path.join(MODELS_DIR, "lstm_model_final.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model dosyasi bulunamadi: {model_path}")
        model = load_model(model_path)
        print(f"[+] LSTM modeli yuklendi: {model_path}")
    elif model_type == "rf":
        model_path = os.path.join(MODELS_DIR, "random_forest_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Random Forest model dosyasi bulunamadi: {model_path}")
        model = joblib.load(model_path)
        print(f"[+] Random Forest modeli yuklendi: {model_path}")
    elif model_type == "nb":
        model_path = os.path.join(MODELS_DIR, "naive_bayes_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Naive Bayes model dosyasi bulunamadi: {model_path}")
        model = joblib.load(model_path)
        print(f"[+] Naive Bayes modeli yuklendi: {model_path}")
    else:
        raise ValueError(f"Gecersiz model tipi: {model_type}")

    return model, scaler


def predict_single(model, scaler, features, model_type="lstm", threshold=0.5):
    """
    Tek bir ag trafigi ornegi icin tahmin yapar.

    Parameters
    ----------
    model : object
        Egitilmis model.
    scaler : StandardScaler
        Normalizasyon nesnesi.
    features : np.ndarray or list
        Ozellik vektoru.
    model_type : str
        Model tipi.
    threshold : float
        Ikili siniflandirma icin esik degeri.

    Returns
    -------
    dict
        Tahmin sonucu.
    """
    features = np.array(features).reshape(1, -1).astype(np.float32)
    features_scaled = scaler.transform(features)

    start_time = time.time()

    if model_type == "lstm":
        features_3d = features_scaled.reshape(1, 1, -1)
        prob = model.predict(features_3d, verbose=0).flatten()[0]
        prediction = int(prob >= threshold)
    else:
        prediction = int(model.predict(features_scaled)[0])
        prob = float(model.predict_proba(features_scaled)[0][1])

    elapsed = time.time() - start_time

    result = {
        "prediction": prediction,
        "label": "SALDIRI" if prediction == 1 else "NORMAL",
        "probability": float(prob),
        "confidence": float(prob) if prediction == 1 else float(1 - prob),
        "elapsed_time_ms": elapsed * 1000,
    }

    return result


def predict_batch(model, scaler, X, model_type="lstm", threshold=0.5):
    """
    Birden fazla ornek icin toplu tahmin yapar.

    Parameters
    ----------
    model : object
        Egitilmis model.
    scaler : StandardScaler
        Normalizasyon nesnesi.
    X : np.ndarray
        Ozellik matrisi.
    model_type : str
        Model tipi.
    threshold : float
        Ikili siniflandirma icin esik degeri.

    Returns
    -------
    dict
        Toplu tahmin sonuclari.
    """
    X_scaled = scaler.transform(X.astype(np.float32))

    start_time = time.time()

    if model_type == "lstm":
        X_3d = X_scaled.reshape(X_scaled.shape[0], 1, -1)
        probs = model.predict(X_3d, verbose=0).flatten()
        predictions = (probs >= threshold).astype(int)
    else:
        predictions = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

    elapsed = time.time() - start_time

    n_attacks = int(np.sum(predictions == 1))
    n_normal = int(np.sum(predictions == 0))

    results = {
        "predictions": predictions,
        "probabilities": probs,
        "total_samples": len(predictions),
        "n_attacks": n_attacks,
        "n_normal": n_normal,
        "attack_ratio": n_attacks / len(predictions) * 100,
        "elapsed_time_ms": elapsed * 1000,
        "avg_time_per_sample_ms": (elapsed * 1000) / len(predictions),
    }

    return results


def simulate_realtime_detection(model, scaler, X, y_true=None,
                                 model_type="lstm", delay=0.1):
    """
    Gercek zamanli saldiri tespitini simule eder.
    Her ornegi teker teker isler ve sonuclari yazdirir.

    Parameters
    ----------
    model : object
        Egitilmis model.
    scaler : StandardScaler
        Normalizasyon nesnesi.
    X : np.ndarray
        Ozellik matrisi.
    y_true : np.ndarray, optional
        Gercek etiketler (dogrulama icin).
    model_type : str
        Model tipi.
    delay : float
        Ornekler arasi bekleme suresi (saniye).
    """
    print("\n" + "=" * 70)
    print("  GERCEK ZAMANLI SALDIRI TESPIT SIMULASYONU")
    print("=" * 70)
    print(f"  Model: {model_type.upper()}")
    print(f"  Toplam ornek: {len(X)}")
    print(f"  Simule gecikme: {delay}s / ornek")
    print("=" * 70)

    attack_count = 0
    normal_count = 0
    correct_count = 0

    for i in range(len(X)):
        result = predict_single(model, scaler, X[i], model_type=model_type)

        status = "🔴 SALDIRI TESPIT EDILDI!" if result["prediction"] == 1 else "🟢 Normal Trafik"

        log_line = (
            f"  [{i + 1:05d}] {status} | "
            f"Guven: {result['confidence']:.2f} | "
            f"Sure: {result['elapsed_time_ms']:.2f}ms"
        )

        if y_true is not None:
            is_correct = result["prediction"] == y_true[i]
            correct_count += int(is_correct)
            log_line += f" | {'DOGRU' if is_correct else 'YANLIS'}"

        print(log_line)

        if result["prediction"] == 1:
            attack_count += 1
        else:
            normal_count += 1

        if delay > 0:
            time.sleep(delay)

    print(f"\n{'─' * 50}")
    print(f"  Toplam: {len(X)} ornek")
    print(f"  Saldiri: {attack_count} ({attack_count / len(X) * 100:.1f}%)")
    print(f"  Normal: {normal_count} ({normal_count / len(X) * 100:.1f}%)")
    if y_true is not None:
        accuracy = correct_count / len(X) * 100
        print(f"  Dogruluk: {accuracy:.2f}%")
    print(f"{'─' * 50}")


def predict_from_csv(csv_path, model_type="lstm"):
    """
    CSV dosyasindan veri okuyup tahmin yapar.

    Parameters
    ----------
    csv_path : str
        CSV dosya yolu.
    model_type : str
        Model tipi.

    Returns
    -------
    pd.DataFrame
        Tahmin sonuclari ile veri.
    """
    print(f"\n[*] CSV dosyasi okunuyor: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Label sutununu ayir (varsa)
    label_col = None
    for col in df.columns:
        if col.strip().lower() == "label":
            label_col = col
            break

    y_true = None
    if label_col:
        y_true = df[label_col].apply(
            lambda x: 0 if str(x).strip() == "BENIGN" else 1
        ).values
        df = df.drop(columns=[label_col])

    # Egitim sirasinda kaldirilan gereksiz sutunlari kaldir
    # (clean_data ile ayni filtreleme)
    columns_to_drop = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in
               ["flow id", "source ip", "destination ip",
                "source port", "destination port", "timestamp"]):
            columns_to_drop.append(col)
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors="ignore")

    # Sayisal olmayan sutunlari kaldir
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].values

    # Sonsuz ve NaN degerleri temizle
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Model ve scaler yukle
    model, scaler = load_prediction_pipeline(model_type=model_type)

    # Tahmin yap
    results = predict_batch(model, scaler, X, model_type=model_type)

    print(f"\n[+] Tahmin Sonuclari:")
    print(f"  Toplam: {results['total_samples']} ornek")
    print(f"  Saldiri: {results['n_attacks']} ({results['attack_ratio']:.1f}%)")
    print(f"  Normal: {results['n_normal']}")
    print(f"  Toplam sure: {results['elapsed_time_ms']:.2f}ms")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ag Saldiri Tespit Sistemi - Tahmin Modulu"
    )
    parser.add_argument(
        "--csv", type=str, help="Tahmin yapilacak CSV dosyasi"
    )
    parser.add_argument(
        "--model", type=str, default="lstm",
        choices=["lstm", "rf", "nb"],
        help="Kullanilacak model (varsayilan: lstm)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Gercek zamanli simulasyon modu"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Simulasyonda kullanilacak ornek sayisi"
    )

    args = parser.parse_args()

    if args.csv:
        predict_from_csv(args.csv, model_type=args.model)
    else:
        print("Kullanim: python predict.py --csv <dosya.csv> --model lstm")
        print("Simulasyon: python predict.py --simulate --n-samples 50")
