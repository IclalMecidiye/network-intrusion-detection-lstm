# Yapay Zeka Tabanlı Siber Saldırı Tespit Sistemi

## LSTM ve Makine Öğrenmesi ile Ağ Saldırı Tespiti

> **Sakarya Uygulamalı Bilimler Üniversitesi - Bilgisayar Mühendisliği Bölümü**  
> **Bitirme Tezi Uygulaması**  
> **İclal MECİDİYE - B200109017**  
> **Danışman: Doç. Dr. Süleyman UZUN**

---

## Proje Hakkında

Bu proje, ağ trafiğini analiz ederek siber saldırıları **gerçek zamanlı** olarak tespit edebilen yapay zeka tabanlı bir siber güvenlik sistemidir. Projede **LSTM (Long Short-Term Memory)** derin öğrenme modeli ana model olarak kullanılmış, **Random Forest** ve **Naive Bayes** algoritmaları ile performans karşılaştırması yapılmıştır.

### Kullanılan Veri Seti

**CICIDS2017** (Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017) veri seti kullanılmaktadır. Bu veri seti aşağıdaki saldırı türlerini içerir:

| Saldırı Türü | Açıklama |
|---|---|
| **DoS / DDoS** | Hizmet dışı bırakma saldırıları (Slowloris, Hulk, GoldenEye) |
| **Brute Force** | Kaba kuvvet saldırıları (FTP-Patator, SSH-Patator) |
| **Web Attack** | Web saldırıları (XSS, SQL Injection, Brute Force) |
| **Infiltration** | Sızma saldırıları |
| **Bot** | Botnet aktiviteleri |
| **PortScan** | Port tarama saldırıları |
| **Heartbleed** | OpenSSL Heartbleed açığı istismarı |

## Proje Yapısı

```
network-intrusion-detection-lstm/
├── config/
│   └── config.py                  # Yapılandırma parametreleri
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Veri ön işleme modülü
│   ├── lstm_model.py              # LSTM model tanımı ve eğitimi
│   ├── comparison_models.py       # Random Forest & Naive Bayes
│   ├── evaluate.py                # Değerlendirme ve görselleştirme
│   ├── train.py                   # Ana eğitim scripti
│   └── predict.py                 # Gerçek zamanlı tahmin modülü
├── app.py                         # Streamlit web arayüzü
├── data/                          # CICIDS2017 veri seti (CSV dosyaları)
├── models/                        # Eğitilmiş modeller
├── outputs/                       # Grafikler ve raporlar
├── requirements.txt               # Python bağımlılıkları
└── README.md                      # Bu dosya
```

## Kurulum

### 1. Gereksinimler

- Python 3.9+
- pip

### 2. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 3. CICIDS2017 Veri Setini İndirin

CICIDS2017 veri setini aşağıdaki linkten indirip `data/` dizinine yerleştirin:

- **İndirme Linki:** https://www.unb.ca/cic/datasets/ids-2017.html
- MachineLearningCSV.zip dosyasını indirip `data/` klasörüne çıkartın

CSV dosyaları `data/` dizininde şu şekilde olmalıdır:
```
data/
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Monday-WorkingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
└── Wednesday-workingHours.pcap_ISCX.csv
```

## Kullanım

### Tüm Modelleri Eğitme

```bash
python src/train.py
```

### Belirli Bir Modeli Eğitme

```bash
# Sadece LSTM
python src/train.py --model lstm

# Sadece Random Forest
python src/train.py --model rf

# Sadece Naive Bayes
python src/train.py --model nb
```

### Hızlı Test (Küçük Örneklem)

```bash
# Her CSV'den 10.000 örnek al
python src/train.py --sample-size 10000
```

### Çoklu Sınıflandırma

```bash
# Saldırı türlerini ayrı ayrı sınıflandır
python src/train.py --multiclass
```

### Web Arayüzü (Streamlit)

```bash
# Web arayüzünü başlatın
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresi otomatik açılacaktır. Web arayüzü ile:
- **CSV dosyası yükleyerek** saldırı tespiti yapabilirsiniz
- **Model eğitimi** başlatabilirsiniz
- **Eğitim sonuçlarını ve grafikleri** görüntüleyebilirsiniz

### Gerçek Zamanlı Tahmin (Terminal)

```bash
# CSV dosyasından tahmin
python src/predict.py --csv data/test_data.csv --model lstm
```

## Model Mimarisi

### LSTM Modeli (Ana Model)

```
Input (1, n_features)
    │
    ▼
LSTM (128 nöron, return_sequences=True)
    │
BatchNormalization
    │
Dropout (0.3)
    │
    ▼
LSTM (64 nöron)
    │
BatchNormalization
    │
Dropout (0.3)
    │
    ▼
Dense (32, ReLU aktivasyon)
    │
Dropout (0.3)
    │
    ▼
Dense (1, Sigmoid) → İkili Sınıflandırma
    veya
Dense (n_classes, Softmax) → Çoklu Sınıflandırma
```

### Hiperparametreler

| Parametre | Değer |
|---|---|
| LSTM Katman 1 | 128 nöron |
| LSTM Katman 2 | 64 nöron |
| Dense Katman | 32 nöron |
| Dropout | 0.3 |
| Öğrenme Oranı | 0.001 |
| Batch Size | 256 |
| Epoch | 50 (Early Stopping ile) |
| Optimizer | Adam |

## Değerlendirme Metrikleri

Proje aşağıdaki metrikleri kullanarak model performansını değerlendirir:

- **Accuracy (Doğruluk):** Doğru sınıflandırma oranı
- **Precision (Kesinlik):** Pozitif tahminlerin ne kadarının doğru olduğu
- **Recall (Duyarlılık):** Gerçek pozitiflerin ne kadarının tespit edildiği
- **F1-Score:** Precision ve Recall'ın harmonik ortalaması
- **Confusion Matrix (Karışıklık Matrisi):** Sınıflandırma hatalarının görselleştirilmesi
- **ROC Eğrisi ve AUC:** Alıcı İşletim Karakteristiği

## Çıktılar

Eğitim sonrasında `outputs/` dizininde aşağıdaki dosyalar oluşturulur:

| Dosya | Açıklama |
|---|---|
| `lstm_training_history.png` | LSTM eğitim loss/accuracy grafikleri |
| `confusion_matrix_lstm.png` | LSTM karışıklık matrisi |
| `confusion_matrix_random_forest.png` | Random Forest karışıklık matrisi |
| `confusion_matrix_naive_bayes.png` | Naive Bayes karışıklık matrisi |
| `roc_curve_lstm.png` | LSTM ROC eğrisi |
| `roc_comparison.png` | Tüm modellerin ROC karşılaştırması |
| `model_comparison.png` | Model performans karşılaştırma grafiği |
| `time_comparison.png` | Eğitim/tahmin süresi karşılaştırması |
| `evaluation_report.txt` | Detaylı değerlendirme raporu |

## Teknolojiler

- **Python 3.9+**
- **TensorFlow / Keras** - LSTM derin öğrenme modeli
- **scikit-learn** - Random Forest, Naive Bayes, değerlendirme metrikleri
- **pandas** - Veri işleme
- **NumPy** - Sayısal hesaplamalar
- **Matplotlib / Seaborn** - Görselleştirme

## Referanslar

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). CICIDS2017 Dataset.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
3. Breiman, L. (2001). Random Forests. Machine Learning.

## Lisans

Bu proje Sakarya Uygulamalı Bilimler Üniversitesi Bilgisayar Mühendisliği Bölümü bitirme tezi kapsamında geliştirilmiştir.
