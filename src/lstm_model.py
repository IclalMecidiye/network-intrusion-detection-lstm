"""
LSTM Model Modulu
=================
Ag saldiri tespiti icin LSTM (Long Short-Term Memory)
derin ogrenme modelini tanimlar, egitir ve degerlendirir.

LSTM, zaman serisi verilerinde uzun vadeli bagimliliklari
ogrenme yetenegine sahiptir. Ag trafigi verilerindeki
ardisik kaliplari tespit etmek icin idealdir.

Model Mimarisi:
    Input -> LSTM(128) -> Dropout(0.3) -> LSTM(64) -> Dropout(0.3)
    -> Dense(32, ReLU) -> Dropout(0.3) -> Dense(output, Sigmoid/Softmax)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS, DROPOUT_RATE,
    LEARNING_RATE, BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE,
    MODELS_DIR
)


def build_lstm_model(n_features, n_classes=2):
    """
    LSTM modelini olusturur.

    Parameters
    ----------
    n_features : int
        Girdi ozellik sayisi.
    n_classes : int
        Sinif sayisi. 2 ise ikili siniflandirma.

    Returns
    -------
    tf.keras.Model
        Derlenims LSTM modeli.
    """
    print("\n" + "=" * 60)
    print("  LSTM Model Mimarisi Olusturuluyor")
    print("=" * 60)

    model = Sequential(name="LSTM_Saldiri_Tespit")

    # Birinci LSTM katmani
    model.add(LSTM(
        units=LSTM_UNITS_1,
        return_sequences=True,
        input_shape=(1, n_features),
        name="lstm_katman_1"
    ))
    model.add(BatchNormalization(name="batch_norm_1"))
    model.add(Dropout(DROPOUT_RATE, name="dropout_1"))

    # Ikinci LSTM katmani
    model.add(LSTM(
        units=LSTM_UNITS_2,
        return_sequences=False,
        name="lstm_katman_2"
    ))
    model.add(BatchNormalization(name="batch_norm_2"))
    model.add(Dropout(DROPOUT_RATE, name="dropout_2"))

    # Dense katman
    model.add(Dense(
        units=DENSE_UNITS,
        activation="relu",
        name="dense_katman"
    ))
    model.add(Dropout(DROPOUT_RATE, name="dropout_3"))

    # Cikis katmani
    if n_classes == 2:
        model.add(Dense(1, activation="sigmoid", name="cikis_katmani"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        model.add(Dense(n_classes, activation="softmax", name="cikis_katmani"))
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    # Model derleme
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.summary()

    return model


def get_callbacks(model_name="lstm_model"):
    """
    Egitim icin callback'leri olusturur.

    Parameters
    ----------
    model_name : str
        Model dosya adi.

    Returns
    -------
    list
        Callback listesi.
    """
    callbacks = []

    # Early Stopping: Dogrulama kaybi iyilesmezse durdur
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Model Checkpoint: En iyi modeli kaydet
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_best.keras")
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    callbacks.append(model_checkpoint)

    # Ogrenme oranini azalt: Dogrulama kaybi durgunlasirsa
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)

    return callbacks


def train_lstm(model, X_train, y_train, X_val, y_val):
    """
    LSTM modelini egitir.

    Parameters
    ----------
    model : tf.keras.Model
        LSTM modeli.
    X_train : np.ndarray
        Egitim verisi (3D: samples, timesteps, features).
    y_train : np.ndarray
        Egitim etiketleri.
    X_val : np.ndarray
        Dogrulama verisi (3D).
    y_val : np.ndarray
        Dogrulama etiketleri.

    Returns
    -------
    tf.keras.callbacks.History
        Egitim gecmisi.
    """
    print("\n" + "=" * 60)
    print("  LSTM Model Egitimi Basladi")
    print("=" * 60)
    print(f"  Epoch: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Ogrenme Orani: {LEARNING_RATE}")
    print(f"  Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print("=" * 60)

    callbacks = get_callbacks()

    # Sinif agirliklarini hesapla (dengesiz veri icin)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_weights = {
        int(cls): total / (len(unique_classes) * count)
        for cls, count in zip(unique_classes, class_counts)
    }
    print(f"\n  Sinif agirliklari: {class_weights}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    print("\n[+] LSTM model egitimi tamamlandi!")

    return history


def save_lstm_model(model, filename="lstm_model_final.keras"):
    """
    Egitilmis LSTM modelini kaydeder.

    Parameters
    ----------
    model : tf.keras.Model
        Kaydedilecek model.
    filename : str
        Dosya adi.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    model.save(filepath)
    print(f"[+] Model kaydedildi: {filepath}")


def load_lstm_model_from_file(filename="lstm_model_final.keras"):
    """
    Kaydedilmis LSTM modelini yukler.

    Parameters
    ----------
    filename : str
        Dosya adi.

    Returns
    -------
    tf.keras.Model
        Yuklenen model.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model dosyasi bulunamadi: {filepath}")
    model = load_model(filepath)
    print(f"[+] Model yuklendi: {filepath}")
    return model


def predict_lstm(model, X, threshold=0.5):
    """
    LSTM modeli ile tahmin yapar.

    Parameters
    ----------
    model : tf.keras.Model
        Egitilmis LSTM modeli.
    X : np.ndarray
        Tahmin yapilacak veri (3D).
    threshold : float
        Ikili siniflandirma icin esik degeri.

    Returns
    -------
    tuple
        (tahminler, olasiliklar)
    """
    probabilities = model.predict(X, verbose=0)

    # Ikili siniflandirma
    if probabilities.shape[1] == 1 or len(probabilities.shape) == 1:
        probabilities = probabilities.flatten()
        predictions = (probabilities >= threshold).astype(int)
    else:
        # Coklu siniflandirma
        predictions = np.argmax(probabilities, axis=1)

    return predictions, probabilities
