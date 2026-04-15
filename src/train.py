"""
Ana Egitim Scripti
==================
Tum modelleri (LSTM, Random Forest, Naive Bayes) egitir,
degerlendirir ve sonuclari gorsellestirir.

Kullanim:
    python src/train.py                          # Tum modelleri egit
    python src/train.py --model lstm             # Sadece LSTM
    python src/train.py --model rf               # Sadece Random Forest
    python src/train.py --model nb               # Sadece Naive Bayes
    python src/train.py --sample-size 50000      # Orneklem ile egit
    python src/train.py --binary                 # Ikili siniflandirma
    python src/train.py --multiclass             # Coklu siniflandirma
"""

import os
import sys
import time
import argparse
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MODELS_DIR, OUTPUTS_DIR
from src.data_preprocessing import prepare_data
from src.lstm_model import (
    build_lstm_model, train_lstm, save_lstm_model, predict_lstm
)
from src.comparison_models import RandomForestModel, NaiveBayesModel
from src.evaluate import (
    calculate_metrics, print_evaluation_report,
    plot_confusion_matrix, plot_roc_curve, plot_training_history,
    plot_model_comparison, plot_time_comparison, plot_roc_comparison,
    generate_full_report
)


def train_and_evaluate_lstm(data):
    """LSTM modelini egitir ve degerlendirir."""
    print("\n" + "#" * 70)
    print("#  LSTM MODEL EGITIMI")
    print("#" * 70)

    # Model olustur
    model = build_lstm_model(
        n_features=data["n_features"],
        n_classes=data["n_classes"]
    )

    # Egit
    start_time = time.time()
    history = train_lstm(
        model,
        data["X_train_lstm"], data["y_train"],
        data["X_val_lstm"], data["y_val"]
    )
    training_time = time.time() - start_time

    # Egitim grafigini kaydet
    plot_training_history(history)

    # Tahmin
    start_time = time.time()
    y_pred, y_prob = predict_lstm(model, data["X_test_lstm"])
    prediction_time = time.time() - start_time

    # Metrikleri hesapla
    metrics = calculate_metrics(data["y_test"], y_pred, data["label_names"])
    metrics["training_time"] = training_time
    metrics["prediction_time"] = prediction_time

    # Rapor
    print_evaluation_report(
        "LSTM", metrics, data["label_names"],
        data["y_test"], y_pred
    )

    # Karisiklik matrisi
    plot_confusion_matrix(
        data["y_test"], y_pred, data["label_names"], "LSTM"
    )

    # ROC egrisi (ikili siniflandirma)
    if data["n_classes"] == 2:
        roc_auc = plot_roc_curve(data["y_test"], y_prob, "LSTM")
        metrics["roc_auc"] = roc_auc

    # Modeli kaydet
    save_lstm_model(model)

    return {
        "metrics": metrics,
        "predictions": y_pred,
        "probabilities": y_prob,
        "history": history,
        "model": model,
    }


def train_and_evaluate_rf(data):
    """Random Forest modelini egitir ve degerlendirir."""
    print("\n" + "#" * 70)
    print("#  RANDOM FOREST MODEL EGITIMI")
    print("#" * 70)

    rf = RandomForestModel()
    rf.train(data["X_train"], data["y_train"])

    # Degerlendir
    result = rf.evaluate(data["X_test"], data["y_test"], data["label_names"])

    # Metrikleri hesapla
    metrics = calculate_metrics(
        data["y_test"], result["predictions"], data["label_names"]
    )
    metrics["training_time"] = result["training_time"]
    metrics["prediction_time"] = result["prediction_time"]

    # Rapor
    print_evaluation_report(
        "Random Forest", metrics, data["label_names"],
        data["y_test"], result["predictions"]
    )

    # Karisiklik matrisi
    plot_confusion_matrix(
        data["y_test"], result["predictions"],
        data["label_names"], "Random Forest"
    )

    # ROC egrisi (ikili siniflandirma)
    if data["n_classes"] == 2:
        y_prob = result["probabilities"][:, 1]
        roc_auc = plot_roc_curve(data["y_test"], y_prob, "Random Forest")
        metrics["roc_auc"] = roc_auc
    else:
        y_prob = None

    # Ozellik onemliligi
    rf.get_feature_importance(data["feature_names"])

    # Modeli kaydet
    rf.save()

    return {
        "metrics": metrics,
        "predictions": result["predictions"],
        "probabilities": y_prob if data["n_classes"] == 2 else result["probabilities"],
        "model": rf,
    }


def train_and_evaluate_nb(data):
    """Naive Bayes modelini egitir ve degerlendirir."""
    print("\n" + "#" * 70)
    print("#  NAIVE BAYES MODEL EGITIMI")
    print("#" * 70)

    nb = NaiveBayesModel()
    nb.train(data["X_train"], data["y_train"])

    # Degerlendir
    result = nb.evaluate(data["X_test"], data["y_test"], data["label_names"])

    # Metrikleri hesapla
    metrics = calculate_metrics(
        data["y_test"], result["predictions"], data["label_names"]
    )
    metrics["training_time"] = result["training_time"]
    metrics["prediction_time"] = result["prediction_time"]

    # Rapor
    print_evaluation_report(
        "Naive Bayes", metrics, data["label_names"],
        data["y_test"], result["predictions"]
    )

    # Karisiklik matrisi
    plot_confusion_matrix(
        data["y_test"], result["predictions"],
        data["label_names"], "Naive Bayes"
    )

    # ROC egrisi (ikili siniflandirma)
    if data["n_classes"] == 2:
        y_prob = result["probabilities"][:, 1]
        roc_auc = plot_roc_curve(data["y_test"], y_prob, "Naive Bayes")
        metrics["roc_auc"] = roc_auc
    else:
        y_prob = None

    # Modeli kaydet
    nb.save()

    return {
        "metrics": metrics,
        "predictions": result["predictions"],
        "probabilities": y_prob if data["n_classes"] == 2 else result["probabilities"],
        "model": nb,
    }


def main():
    """Ana egitim fonksiyonu."""
    parser = argparse.ArgumentParser(
        description="Yapay Zeka Tabanli Siber Saldiri Tespit Sistemi - Egitim"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "lstm", "rf", "nb"],
        help="Egitilecek model (varsayilan: all)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Her CSV dosyasindan alinacak ornek sayisi (hizli test icin)"
    )
    parser.add_argument(
        "--binary", action="store_true", default=True,
        help="Ikili siniflandirma (Normal/Saldiri)"
    )
    parser.add_argument(
        "--multiclass", action="store_true",
        help="Coklu siniflandirma (saldiri turleri)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Veri dizini yolu"
    )

    args = parser.parse_args()

    binary = not args.multiclass

    print("\n" + "=" * 70)
    print("  YAPAY ZEKA TABANLI SIBER SALDIRI TESPIT SISTEMI")
    print("  LSTM ve Makine Ogrenmesi ile Ag Saldiri Tespiti")
    print("=" * 70)
    print(f"  Siniflandirma: {'Ikili (Normal/Saldiri)' if binary else 'Coklu (Saldiri Turleri)'}")
    print(f"  Model: {args.model}")
    if args.sample_size:
        print(f"  Orneklem boyutu: {args.sample_size}")
    print("=" * 70)

    # ============================================================
    # 1. Veri On Isleme
    # ============================================================
    data = prepare_data(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        binary=binary
    )

    # Scaler'i kaydet (tahmin modulu icin)
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(data["scaler"], scaler_path)
    print(f"\n[+] Scaler kaydedildi: {scaler_path}")

    # ============================================================
    # 2. Model Egitimi ve Degerlendirme
    # ============================================================
    all_results = {}
    model_probs = {}

    # LSTM
    if args.model in ["all", "lstm"]:
        lstm_result = train_and_evaluate_lstm(data)
        all_results["LSTM"] = lstm_result["metrics"]
        if data["n_classes"] == 2:
            model_probs["LSTM"] = lstm_result["probabilities"]

    # Random Forest
    if args.model in ["all", "rf"]:
        rf_result = train_and_evaluate_rf(data)
        all_results["Random Forest"] = rf_result["metrics"]
        if data["n_classes"] == 2 and rf_result["probabilities"] is not None:
            model_probs["Random Forest"] = rf_result["probabilities"]

    # Naive Bayes
    if args.model in ["all", "nb"]:
        nb_result = train_and_evaluate_nb(data)
        all_results["Naive Bayes"] = nb_result["metrics"]
        if data["n_classes"] == 2 and nb_result["probabilities"] is not None:
            model_probs["Naive Bayes"] = nb_result["probabilities"]

    # ============================================================
    # 3. Karsilastirma Grafikleri
    # ============================================================
    if len(all_results) > 1:
        print("\n" + "#" * 70)
        print("#  MODEL KARSILASTIRMASI")
        print("#" * 70)

        # Performans karsilastirma
        plot_model_comparison(all_results)

        # Sure karsilastirma
        plot_time_comparison(all_results)

        # ROC karsilastirma (ikili siniflandirma)
        if data["n_classes"] == 2 and len(model_probs) > 1:
            plot_roc_comparison(data["y_test"], model_probs)

    # ============================================================
    # 4. Genel Rapor
    # ============================================================
    generate_full_report(all_results, data["y_test"])

    # ============================================================
    # 5. Sonuc Ozeti
    # ============================================================
    print("\n" + "=" * 70)
    print("  EGITIM TAMAMLANDI!")
    print("=" * 70)

    if all_results:
        best_model = max(all_results.items(), key=lambda x: x[1]["accuracy"])
        print(f"\n  En iyi model: {best_model[0]}")
        print(f"  Dogruluk: {best_model[1]['accuracy'] * 100:.2f}%")
        print(f"  F1-Score: {best_model[1]['f1_score'] * 100:.2f}%")

    print(f"\n  Modeller '{MODELS_DIR}' dizinine kaydedildi.")
    print(f"  Grafikler '{OUTPUTS_DIR}' dizinine kaydedildi.")
    print("=" * 70)


if __name__ == "__main__":
    main()
