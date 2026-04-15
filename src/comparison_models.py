"""
Karsilastirma Modelleri Modulu
===============================
LSTM modeli ile karsilastirma yapmak icin
Random Forest ve Naive Bayes modellerini tanimlar.

Tezdeki gibi, bu algoritmalar CICIDS2017 veri seti
uzerinde egitilir ve performanslari LSTM ile karsilastirilir.
"""

import os
import time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT,
    NB_VAR_SMOOTHING, RANDOM_STATE, MODELS_DIR
)


class RandomForestModel:
    """
    Random Forest siniflandirici modeli.

    Random Forest, birden fazla karar agacindan olusan bir
    topluluk ogrenme yontemidir. Yuksek dogruluk orani,
    dayanikliligi ve siniflandirma gucu nedeniyle tercih edilir.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        self.training_time = 0
        self.prediction_time = 0

    def train(self, X_train, y_train):
        """Random Forest modelini egitir."""
        print("\n" + "=" * 60)
        print("  Random Forest Egitimi Basladi")
        print("=" * 60)
        print(f"  Agac sayisi: {RF_N_ESTIMATORS}")
        print(f"  Maks derinlik: {RF_MAX_DEPTH}")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        print(f"\n[+] Random Forest egitimi tamamlandi!")
        print(f"  Egitim suresi: {self.training_time:.2f} saniye")

        return self

    def predict(self, X):
        """Tahmin yapar."""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = time.time() - start_time
        return predictions

    def predict_proba(self, X):
        """Olasilik tahminleri yapar."""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, label_names=None):
        """Model performansini degerlendirir."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"\n  Random Forest Dogruluk: {accuracy * 100:.2f}%")
        print(f"  Tahmin suresi: {self.prediction_time:.4f} saniye")

        if label_names:
            print("\n  Siniflandirma Raporu:")
            print(classification_report(y_test, predictions,
                                        target_names=label_names))

        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "probabilities": self.predict_proba(X_test),
            "training_time": self.training_time,
            "prediction_time": self.prediction_time,
        }

    def save(self, filename="random_forest_model.joblib"):
        """Modeli kaydeder."""
        filepath = os.path.join(MODELS_DIR, filename)
        joblib.dump(self.model, filepath)
        print(f"[+] Random Forest modeli kaydedildi: {filepath}")

    def load(self, filename="random_forest_model.joblib"):
        """Modeli yukler."""
        filepath = os.path.join(MODELS_DIR, filename)
        self.model = joblib.load(filepath)
        print(f"[+] Random Forest modeli yuklendi: {filepath}")
        return self

    def get_feature_importance(self, feature_names):
        """Ozellik onemliligi degerlerini dondurur."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\n  En onemli 20 ozellik:")
        for i in range(min(20, len(feature_names))):
            idx = indices[i]
            print(f"    {i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

        return importances, indices


class NaiveBayesModel:
    """
    Gaussian Naive Bayes siniflandirici modeli.

    Naive Bayes, olasiliksal bir siniflandirma algoritmasidir.
    Hizli islem suresi ve basit yapisi nedeniyle tercih edilir.
    """

    def __init__(self):
        self.model = GaussianNB(var_smoothing=NB_VAR_SMOOTHING)
        self.training_time = 0
        self.prediction_time = 0

    def train(self, X_train, y_train):
        """Naive Bayes modelini egitir."""
        print("\n" + "=" * 60)
        print("  Naive Bayes Egitimi Basladi")
        print("=" * 60)

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        print(f"\n[+] Naive Bayes egitimi tamamlandi!")
        print(f"  Egitim suresi: {self.training_time:.2f} saniye")

        return self

    def predict(self, X):
        """Tahmin yapar."""
        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = time.time() - start_time
        return predictions

    def predict_proba(self, X):
        """Olasilik tahminleri yapar."""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, label_names=None):
        """Model performansini degerlendirir."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"\n  Naive Bayes Dogruluk: {accuracy * 100:.2f}%")
        print(f"  Tahmin suresi: {self.prediction_time:.4f} saniye")

        if label_names:
            print("\n  Siniflandirma Raporu:")
            print(classification_report(y_test, predictions,
                                        target_names=label_names))

        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "probabilities": self.predict_proba(X_test),
            "training_time": self.training_time,
            "prediction_time": self.prediction_time,
        }

    def save(self, filename="naive_bayes_model.joblib"):
        """Modeli kaydeder."""
        filepath = os.path.join(MODELS_DIR, filename)
        joblib.dump(self.model, filepath)
        print(f"[+] Naive Bayes modeli kaydedildi: {filepath}")

    def load(self, filename="naive_bayes_model.joblib"):
        """Modeli yukler."""
        filepath = os.path.join(MODELS_DIR, filename)
        self.model = joblib.load(filepath)
        print(f"[+] Naive Bayes modeli yuklendi: {filepath}")
        return self
