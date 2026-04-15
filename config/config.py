"""
Proje yapilandirma dosyasi.
Tum model parametreleri, dosya yollari ve sabitleri burada tanimliyoruz.
"""

import os

# ============================================================
# Dosya Yollari
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Dizinleri olustur
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================
# CICIDS2017 Veri Seti
# ============================================================
# CICIDS2017 veri seti CSV dosya isimleri
CICIDS2017_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

# Veri seti indirme URL'si
CICIDS2017_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"

# ============================================================
# Veri On Isleme Parametreleri
# ============================================================
TEST_SIZE = 0.2          # Test verisi orani
VALIDATION_SIZE = 0.1    # Dogrulama verisi orani
RANDOM_STATE = 42        # Tekrarlanabilirlik icin sabit seed

# Siniflandirma: Ikili (Binary) vs Coklu (Multi-class)
BINARY_CLASSIFICATION = True  # True: Normal/Saldiri, False: Saldiri turleri

# Saldiri turlerinin etiket eslesmesi
ATTACK_LABELS = {
    "BENIGN": 0,
    "FTP-Patator": 1,
    "SSH-Patator": 1,
    "DoS slowloris": 1,
    "DoS Slowhttptest": 1,
    "DoS Hulk": 1,
    "DoS GoldenEye": 1,
    "Heartbleed": 1,
    "Web Attack – Brute Force": 1,
    "Web Attack – XSS": 1,
    "Web Attack – Sql Injection": 1,
    "Infiltration": 1,
    "Bot": 1,
    "PortScan": 1,
    "DDoS": 1,
}

# Coklu siniflandirma etiketleri
MULTICLASS_LABELS = {
    "BENIGN": 0,
    "FTP-Patator": 1,
    "SSH-Patator": 2,
    "DoS slowloris": 3,
    "DoS Slowhttptest": 3,
    "DoS Hulk": 3,
    "DoS GoldenEye": 3,
    "Heartbleed": 4,
    "Web Attack – Brute Force": 5,
    "Web Attack – XSS": 5,
    "Web Attack – Sql Injection": 5,
    "Infiltration": 6,
    "Bot": 7,
    "PortScan": 8,
    "DDoS": 9,
}

MULTICLASS_NAMES = {
    0: "Normal (BENIGN)",
    1: "FTP-Patator",
    2: "SSH-Patator",
    3: "DoS",
    4: "Heartbleed",
    5: "Web Attack",
    6: "Infiltration",
    7: "Bot",
    8: "PortScan",
    9: "DDoS",
}

# ============================================================
# LSTM Model Parametreleri
# ============================================================
LSTM_UNITS_1 = 128          # Birinci LSTM katmani noron sayisi
LSTM_UNITS_2 = 64           # Ikinci LSTM katmani noron sayisi
DENSE_UNITS = 32            # Dense katman noron sayisi
DROPOUT_RATE = 0.3          # Dropout orani (overfitting onleme)
LEARNING_RATE = 0.001       # Ogrenme orani
BATCH_SIZE = 256             # Batch boyutu
EPOCHS = 50                 # Egitim epoch sayisi
EARLY_STOPPING_PATIENCE = 5 # Early stopping icin sabir

# ============================================================
# Random Forest Parametreleri
# ============================================================
RF_N_ESTIMATORS = 100       # Agac sayisi
RF_MAX_DEPTH = 20           # Maksimum derinlik
RF_MIN_SAMPLES_SPLIT = 5    # Minimum bolme ornegi

# ============================================================
# Naive Bayes Parametreleri
# ============================================================
NB_VAR_SMOOTHING = 1e-9     # Gaussian Naive Bayes smoothing

# ============================================================
# Cikti ve Gorsellestime
# ============================================================
FIGURE_DPI = 150             # Grafik cozunurlugu
FIGURE_SIZE = (12, 8)        # Grafik boyutu
CONFUSION_MATRIX_FIGSIZE = (10, 8)
ROC_CURVE_FIGSIZE = (10, 8)
