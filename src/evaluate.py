"""
Degerlendirme ve Gorsellestime Modulu
======================================
Model performansini degerlendirmek ve sonuclari
gorsellestirmek icin fonksiyonlar icerir.

Metrikler:
- Accuracy (Dogruluk)
- Precision (Kesinlik)
- Recall (Duyarlilik)
- F1-Score
- Confusion Matrix (Karisiklik Matrisi)
- ROC Curve ve AUC (Alici Isletim Karakteristigi)
- Model karsilastirma grafikleri
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    OUTPUTS_DIR, FIGURE_DPI, FIGURE_SIZE,
    CONFUSION_MATRIX_FIGSIZE, ROC_CURVE_FIGSIZE
)


def calculate_metrics(y_true, y_pred, label_names=None):
    """
    Tum degerlendirme metriklerini hesaplar.

    Parameters
    ----------
    y_true : np.ndarray
        Gercek etiketler.
    y_pred : np.ndarray
        Tahmin edilen etiketler.
    label_names : list, optional
        Sinif isimleri.

    Returns
    -------
    dict
        Metrikler sozlugu.
    """
    n_classes = len(np.unique(y_true))
    average = "binary" if n_classes == 2 else "weighted"

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    return metrics


def print_evaluation_report(model_name, metrics, label_names=None, y_true=None, y_pred=None):
    """
    Degerlendirme raporunu yazdirir.

    Parameters
    ----------
    model_name : str
        Model adi.
    metrics : dict
        Metrikler sozlugu.
    label_names : list, optional
        Sinif isimleri.
    y_true : np.ndarray, optional
        Gercek etiketler (classification_report icin).
    y_pred : np.ndarray, optional
        Tahmin edilen etiketler.
    """
    print(f"\n{'=' * 60}")
    print(f"  {model_name} - Degerlendirme Raporu")
    print(f"{'=' * 60}")
    print(f"  Dogruluk (Accuracy):   {metrics['accuracy'] * 100:.2f}%")
    print(f"  Kesinlik (Precision):  {metrics['precision'] * 100:.2f}%")
    print(f"  Duyarlilik (Recall):   {metrics['recall'] * 100:.2f}%")
    print(f"  F1-Score:              {metrics['f1_score'] * 100:.2f}%")

    if y_true is not None and y_pred is not None:
        print(f"\n  Detayli Siniflandirma Raporu:")
        if label_names:
            print(classification_report(y_true, y_pred, target_names=label_names))
        else:
            print(classification_report(y_true, y_pred))

    print(f"{'=' * 60}")


def plot_confusion_matrix(y_true, y_pred, label_names, model_name,
                          save_path=None, normalize=True):
    """
    Karisiklik matrisini cizer ve kaydeder.

    Parameters
    ----------
    y_true : np.ndarray
        Gercek etiketler.
    y_pred : np.ndarray
        Tahmin edilen etiketler.
    label_names : list
        Sinif isimleri.
    model_name : str
        Model adi.
    save_path : str, optional
        Kayit yolu.
    normalize : bool
        Normalize edilmis matris mi?
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2%"
        title = f"{model_name} - Normalize Karisiklik Matrisi"
    else:
        cm_display = cm
        fmt = "d"
        title = f"{model_name} - Karisiklik Matrisi"

    fig, ax = plt.subplots(figsize=CONFUSION_MATRIX_FIGSIZE)
    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=ax, square=True, linewidths=0.5
    )
    ax.set_xlabel("Tahmin Edilen Sinif", fontsize=12)
    ax.set_ylabel("Gercek Sinif", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path is None:
        safe_name = model_name.lower().replace(" ", "_")
        save_path = os.path.join(OUTPUTS_DIR, f"confusion_matrix_{safe_name}.png")

    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] Karisiklik matrisi kaydedildi: {save_path}")


def plot_roc_curve(y_true, y_prob, model_name, save_path=None):
    """
    ROC egrisini cizer ve AUC degerini hesaplar (ikili siniflandirma).

    Parameters
    ----------
    y_true : np.ndarray
        Gercek etiketler.
    y_prob : np.ndarray
        Pozitif sinif olasiliklari.
    model_name : str
        Model adi.
    save_path : str, optional
        Kayit yolu.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=ROC_CURVE_FIGSIZE)
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC Egrisi (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--",
            label="Rastgele Siniflandirici")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Yanlis Pozitif Orani (FPR)", fontsize=12)
    ax.set_ylabel("Dogru Pozitif Orani (TPR)", fontsize=12)
    ax.set_title(f"{model_name} - ROC Egrisi", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        safe_name = model_name.lower().replace(" ", "_")
        save_path = os.path.join(OUTPUTS_DIR, f"roc_curve_{safe_name}.png")

    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] ROC egrisi kaydedildi: {save_path}")

    return roc_auc


def plot_training_history(history, save_path=None):
    """
    LSTM egitim gecmisini (loss ve accuracy) gorsellestirir.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Egitim gecmisi.
    save_path : str, optional
        Kayit yolu.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss grafigi
    axes[0].plot(history.history["loss"], label="Egitim Kaybi", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Dogrulama Kaybi", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Kayip (Loss)", fontsize=12)
    axes[0].set_title("Model Kayip Grafigi", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy grafigi
    axes[1].plot(history.history["accuracy"], label="Egitim Dogrulugu", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Dogrulama Dogrulugu", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Dogruluk (Accuracy)", fontsize=12)
    axes[1].set_title("Model Dogruluk Grafigi", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "lstm_training_history.png")

    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] Egitim gecmisi grafigi kaydedildi: {save_path}")


def plot_model_comparison(results, save_path=None):
    """
    Tum modellerin performans karsilastirma grafiklerini olusturur.

    Parameters
    ----------
    results : dict
        Model sonuclari. Format:
        {model_adi: {"accuracy": ..., "precision": ..., ...}}
    save_path : str, optional
        Kayit yolu.
    """
    model_names = list(results.keys())
    metrics_names = ["Dogruluk\n(Accuracy)", "Kesinlik\n(Precision)",
                     "Duyarlilik\n(Recall)", "F1-Score"]
    metrics_keys = ["accuracy", "precision", "recall", "f1_score"]

    values = {key: [] for key in metrics_keys}
    for model_name in model_names:
        for key in metrics_keys:
            values[key].append(results[model_name].get(key, 0))

    x = np.arange(len(metrics_names))
    width = 0.25
    n_models = len(model_names)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, model_name in enumerate(model_names):
        model_values = [values[key][i] * 100 for key in metrics_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, model_values, width,
                       label=model_name, color=colors[i % len(colors)],
                       edgecolor="white", linewidth=0.5)

        # Deger etiketlerini ekle
        for bar, val in zip(bars, model_values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

    ax.set_xlabel("Metrikler", fontsize=13)
    ax.set_ylabel("Deger (%)", fontsize=13)
    ax.set_title("Model Performans Karsilastirmasi", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")

    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] Model karsilastirma grafigi kaydedildi: {save_path}")


def plot_time_comparison(results, save_path=None):
    """
    Modellerin egitim ve tahmin suresi karsilastirmasini gorsellestirir.

    Parameters
    ----------
    results : dict
        Model sonuclari.
    save_path : str, optional
        Kayit yolu.
    """
    model_names = list(results.keys())
    training_times = [results[m].get("training_time", 0) for m in model_names]
    prediction_times = [results[m].get("prediction_time", 0) for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    # Egitim suresi
    bars1 = axes[0].bar(model_names, training_times,
                        color=colors[:len(model_names)],
                        edgecolor="white", linewidth=0.5)
    axes[0].set_ylabel("Sure (saniye)", fontsize=12)
    axes[0].set_title("Egitim Suresi Karsilastirmasi", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, training_times):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                     f"{val:.2f}s", ha="center", va="bottom", fontsize=10,
                     fontweight="bold")

    # Tahmin suresi
    bars2 = axes[1].bar(model_names, prediction_times,
                        color=colors[:len(model_names)],
                        edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("Sure (saniye)", fontsize=12)
    axes[1].set_title("Tahmin Suresi Karsilastirmasi", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, prediction_times):
        axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                     f"{val:.4f}s", ha="center", va="bottom", fontsize=10,
                     fontweight="bold")

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "time_comparison.png")

    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] Sure karsilastirma grafigi kaydedildi: {save_path}")


def plot_roc_comparison(y_true, model_probs, save_path=None):
    """
    Birden fazla modelin ROC egrisini ayni grafik uzerinde cizer.

    Parameters
    ----------
    y_true : np.ndarray
        Gercek etiketler.
    model_probs : dict
        Model olasiliklari. Format: {model_adi: olasiliklar}
    save_path : str, optional
        Kayit yolu.
    """
    fig, ax = plt.subplots(figsize=ROC_CURVE_FIGSIZE)

    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, (model_name, y_prob) in enumerate(model_probs.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{model_name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--",
            label="Rastgele Siniflandirici")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Yanlis Pozitif Orani (FPR)", fontsize=12)
    ax.set_ylabel("Dogru Pozitif Orani (TPR)", fontsize=12)
    ax.set_title("ROC Egrisi Karsilastirmasi", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "roc_comparison.png")

    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] ROC karsilastirma grafigi kaydedildi: {save_path}")


def generate_full_report(results, y_true, save_path=None):
    """
    Tum modeller icin kapsamli bir degerlendirme raporu olusturur
    ve metin dosyasina kaydeder.

    Parameters
    ----------
    results : dict
        Model sonuclari.
    y_true : np.ndarray
        Gercek etiketler.
    save_path : str, optional
        Rapor kayit yolu.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "evaluation_report.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("  YAPAY ZEKA TABANLI SIBER SALDIRI TESPIT SISTEMI")
    lines.append("  Degerlendirme Raporu")
    lines.append("=" * 70)
    lines.append("")

    for model_name, result in results.items():
        lines.append(f"\n{'─' * 50}")
        lines.append(f"  Model: {model_name}")
        lines.append(f"{'─' * 50}")
        lines.append(f"  Dogruluk (Accuracy):   {result['accuracy'] * 100:.2f}%")
        lines.append(f"  Kesinlik (Precision):  {result['precision'] * 100:.2f}%")
        lines.append(f"  Duyarlilik (Recall):   {result['recall'] * 100:.2f}%")
        lines.append(f"  F1-Score:              {result['f1_score'] * 100:.2f}%")
        if "training_time" in result:
            lines.append(f"  Egitim Suresi:         {result['training_time']:.2f} saniye")
        if "prediction_time" in result:
            lines.append(f"  Tahmin Suresi:         {result['prediction_time']:.4f} saniye")

    # En iyi model
    lines.append(f"\n\n{'=' * 70}")
    lines.append("  EN IYI MODEL")
    lines.append(f"{'=' * 70}")

    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
    lines.append(f"  {best_model[0]} - Dogruluk: {best_model[1]['accuracy'] * 100:.2f}%")
    lines.append(f"{'=' * 70}")

    report_text = "\n".join(lines)
    print(report_text)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  [+] Rapor kaydedildi: {save_path}")

    return report_text
