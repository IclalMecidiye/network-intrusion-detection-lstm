"""
Streamlit Web Arayuzu
=====================
Yapay Zeka Tabanli Siber Saldiri Tespit Sistemi icin
kullanici dostu web arayuzu.

Calistirma:
    streamlit run app.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Proje yolunu ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import (
    MODELS_DIR, OUTPUTS_DIR, DATA_DIR,
    ATTACK_LABELS, MULTICLASS_LABELS, MULTICLASS_NAMES
)

# ============================================================
# Sayfa Yapilandirmasi
# ============================================================
st.set_page_config(
    page_title="Siber Saldiri Tespit Sistemi",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS Stilleri
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
        opacity: 0.9;
    }
    .metric-card h1 {
        font-size: 2rem;
        margin: 0;
        font-weight: bold;
    }
    .attack-alert {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    .normal-alert {
        background-color: #00C851;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Yardimci Fonksiyonlar
# ============================================================
@st.cache_resource
def load_model_and_scaler(model_type):
    """Modeli ve scaler'i yukler (cache'lenir)."""
    import joblib

    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    if not os.path.exists(scaler_path):
        return None, None, "Scaler dosyasi bulunamadi. Lutfen once modeli egitin."

    scaler = joblib.load(scaler_path)

    if model_type == "LSTM":
        model_path = os.path.join(MODELS_DIR, "lstm_model_final.keras")
        if not os.path.exists(model_path):
            return None, None, "LSTM model dosyasi bulunamadi."
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
    elif model_type == "Random Forest":
        model_path = os.path.join(MODELS_DIR, "random_forest_model.joblib")
        if not os.path.exists(model_path):
            return None, None, "Random Forest model dosyasi bulunamadi."
        model = joblib.load(model_path)
    elif model_type == "Naive Bayes":
        model_path = os.path.join(MODELS_DIR, "naive_bayes_model.joblib")
        if not os.path.exists(model_path):
            return None, None, "Naive Bayes model dosyasi bulunamadi."
        model = joblib.load(model_path)
    else:
        return None, None, "Gecersiz model tipi."

    return model, scaler, None


def preprocess_uploaded_csv(df):
    """Yuklenen CSV dosyasini tahmin icin hazirlar."""
    df.columns = df.columns.str.strip()

    # Label sutununu ayir (varsa)
    label_col = None
    y_true = None
    for col in df.columns:
        if col.strip().lower() == "label":
            label_col = col
            break

    if label_col:
        y_true = df[label_col].apply(
            lambda x: 0 if str(x).strip() == "BENIGN" else 1
        ).values
        label_values = df[label_col].values
        df = df.drop(columns=[label_col])
    else:
        label_values = None

    # Egitim sirasinda kaldirilan gereksiz sutunlari kaldir
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

    return X, y_true, label_values


def predict_with_model(model, scaler, X, model_type, threshold=0.5):
    """Model ile tahmin yapar."""
    X_scaled = scaler.transform(X.astype(np.float32))

    start_time = time.time()

    if model_type == "LSTM":
        X_3d = X_scaled.reshape(X_scaled.shape[0], 1, -1)
        probs = model.predict(X_3d, verbose=0).flatten()
        predictions = (probs >= threshold).astype(int)
    else:
        predictions = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

    elapsed = time.time() - start_time

    return predictions, probs, elapsed


def create_confusion_matrix_fig(y_true, y_pred, labels=None):
    """Karisiklik matrisi grafigi olusturur."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    if labels is None:
        labels = ["Normal", "Saldiri"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Normalize edilmis
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=axes[0], square=True, linewidths=0.5)
    axes[0].set_xlabel("Tahmin Edilen", fontsize=11)
    axes[0].set_ylabel("Gercek", fontsize=11)
    axes[0].set_title("Normalize Karisiklik Matrisi", fontsize=13, fontweight="bold")

    # Ham sayilar
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels, yticklabels=labels,
                ax=axes[1], square=True, linewidths=0.5)
    axes[1].set_xlabel("Tahmin Edilen", fontsize=11)
    axes[1].set_ylabel("Gercek", fontsize=11)
    axes[1].set_title("Karisiklik Matrisi (Sayi)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


def create_roc_curve_fig(y_true, y_prob, model_name):
    """ROC egrisi grafigi olusturur."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#667eea", lw=2.5,
            label=f"ROC Egrisi (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--",
            label="Rastgele Siniflandirici")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#667eea")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Yanlis Pozitif Orani (FPR)", fontsize=12)
    ax.set_ylabel("Dogru Pozitif Orani (TPR)", fontsize=12)
    ax.set_title(f"{model_name} - ROC Egrisi", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, roc_auc


def create_prediction_distribution_fig(predictions, probs):
    """Tahmin dagilimi grafigi olusturur."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pasta grafigi
    n_normal = int(np.sum(predictions == 0))
    n_attack = int(np.sum(predictions == 1))
    colors_pie = ["#00C851", "#ff4444"]
    explode = (0, 0.05)

    axes[0].pie(
        [n_normal, n_attack],
        labels=["Normal", "Saldiri"],
        autopct="%1.1f%%",
        colors=colors_pie,
        explode=explode,
        shadow=True,
        startangle=90,
        textprops={"fontsize": 12}
    )
    axes[0].set_title("Sinif Dagilimi", fontsize=13, fontweight="bold")

    # Olasilik dagilimi
    axes[1].hist(probs, bins=50, color="#667eea", edgecolor="white",
                 alpha=0.8)
    axes[1].axvline(x=0.5, color="red", linestyle="--", lw=2,
                    label="Esik Degeri (0.5)")
    axes[1].set_xlabel("Saldiri Olasiligi", fontsize=12)
    axes[1].set_ylabel("Frekans", fontsize=12)
    axes[1].set_title("Olasilik Dagilimi", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def get_available_models():
    """Mevcut egitilmis modelleri kontrol eder."""
    models = {}
    if os.path.exists(os.path.join(MODELS_DIR, "lstm_model_final.keras")):
        models["LSTM"] = True
    if os.path.exists(os.path.join(MODELS_DIR, "random_forest_model.joblib")):
        models["Random Forest"] = True
    if os.path.exists(os.path.join(MODELS_DIR, "naive_bayes_model.joblib")):
        models["Naive Bayes"] = True
    return models


def get_output_images():
    """Outputs klasorundeki grafikleri listeler."""
    images = {}
    if os.path.exists(OUTPUTS_DIR):
        for f in sorted(os.listdir(OUTPUTS_DIR)):
            if f.endswith(".png"):
                images[f] = os.path.join(OUTPUTS_DIR, f)
    return images


# ============================================================
# Ana Sayfa Baslik
# ============================================================
st.markdown('<div class="main-header">🛡️ Yapay Zeka Tabanli Siber Saldiri Tespit Sistemi</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">LSTM ve Makine Ogrenmesi ile Ag Trafigi Analizi | CICIDS2017 Veri Seti</div>',
            unsafe_allow_html=True)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/cyber-security.png", width=80)
    st.markdown("### ⚙️ Ayarlar")
    st.markdown("---")

    # Mevcut modelleri kontrol et
    available_models = get_available_models()

    if available_models:
        st.success(f"✅ {len(available_models)} model hazir")
        for model_name in available_models:
            st.markdown(f"  - {model_name}")
    else:
        st.warning("⚠️ Henuz egitilmis model yok. Lutfen once 'Model Egitimi' sekmesinden egitim yapin.")

    st.markdown("---")
    st.markdown("### 📊 Proje Bilgileri")
    st.markdown("""
    - **Veri Seti:** CICIDS2017
    - **Ana Model:** LSTM
    - **Karsilastirma:** RF, NB
    - **Dil:** Python / TensorFlow
    """)

    st.markdown("---")
    st.markdown("### 📁 Dizin Yapisi")
    st.code(f"Modeller: {MODELS_DIR}\nCiktilar: {OUTPUTS_DIR}\nVeri: {DATA_DIR}", language=None)

# ============================================================
# Ana Sekmeler
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Saldiri Tespiti",
    "🏋️ Model Egitimi",
    "📈 Sonuclar ve Grafikler",
    "ℹ️ Hakkinda"
])

# ============================================================
# Sekme 1: Saldiri Tespiti
# ============================================================
with tab1:
    st.markdown("## 🔍 Ag Trafigi Analizi ve Saldiri Tespiti")
    st.markdown("CSV formatindaki ag trafigi verinizi yukleyin, yapay zeka modeli ile analiz edin.")

    col_upload, col_settings = st.columns([2, 1])

    with col_settings:
        st.markdown("### ⚙️ Analiz Ayarlari")

        if available_models:
            selected_model = st.selectbox(
                "Model Secimi:",
                list(available_models.keys()),
                help="Tahmin icin kullanilacak model"
            )
        else:
            st.error("Egitilmis model bulunamadi!")
            selected_model = None

        threshold = st.slider(
            "Esik Degeri (Threshold):",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Bu degerden yuksek olasiliklar 'Saldiri' olarak siniflandirilir"
        )

        st.markdown("---")
        st.markdown("### 📋 CSV Format Bilgisi")
        st.markdown("""
        CSV dosyaniz CICIDS2017 formatinda olmalidir:
        - **Label** sutunu (varsa): Gercek etiketler
        - **Ozellik sutunlari**: Ag trafigi ozellikleri
        - Gereksiz sutunlar (IP, Port vb.) otomatik kaldirilir
        """)

    with col_upload:
        uploaded_file = st.file_uploader(
            "📂 CSV Dosyasi Yukleyin",
            type=["csv"],
            help="CICIDS2017 formatinda ag trafigi verisi"
        )

        if uploaded_file is not None and selected_model:
            # Dosya bilgileri
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"📄 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")

            # Veriyi oku
            with st.spinner("Veri okunuyor..."):
                df_raw = pd.read_csv(uploaded_file, low_memory=False)

            st.success(f"✅ {len(df_raw)} satir, {len(df_raw.columns)} sutun yuklendi")

            # Veri on izleme
            with st.expander("📋 Veri On Izleme (ilk 5 satir)", expanded=False):
                st.dataframe(df_raw.head(), use_container_width=True)

            # Analiz butonu
            if st.button("🚀 Analizi Baslat", type="primary", use_container_width=True):
                with st.spinner(f"{selected_model} modeli yukleniyor..."):
                    model, scaler, error = load_model_and_scaler(selected_model)

                if error:
                    st.error(f"❌ Hata: {error}")
                else:
                    # Veriyi hazirla
                    with st.spinner("Veri on isleniyor..."):
                        X, y_true, label_values = preprocess_uploaded_csv(df_raw)

                    # Ozellik sayisi kontrolu
                    expected_features = scaler.n_features_in_
                    if X.shape[1] != expected_features:
                        st.error(
                            f"❌ Ozellik sayisi uyusmazligi! "
                            f"Beklenen: {expected_features}, Gelen: {X.shape[1]}. "
                            f"Lutfen CICIDS2017 formatinda veri yukleyin."
                        )
                    else:
                        # Tahmin yap
                        with st.spinner(f"{selected_model} ile tahmin yapiliyor..."):
                            predictions, probs, elapsed = predict_with_model(
                                model, scaler, X, selected_model, threshold
                            )

                        # ========== SONUCLAR ==========
                        st.markdown("---")
                        st.markdown("## 📊 Analiz Sonuclari")

                        n_total = len(predictions)
                        n_attack = int(np.sum(predictions == 1))
                        n_normal = int(np.sum(predictions == 0))
                        attack_ratio = n_attack / n_total * 100

                        # Metrik kartlari
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Toplam Ornek", f"{n_total:,}")
                        with col2:
                            st.metric("Normal Trafik", f"{n_normal:,}",
                                      delta=f"{100 - attack_ratio:.1f}%")
                        with col3:
                            st.metric("Saldiri Tespiti", f"{n_attack:,}",
                                      delta=f"{attack_ratio:.1f}%",
                                      delta_color="inverse")
                        with col4:
                            st.metric("Analiz Suresi", f"{elapsed:.2f}s")

                        # Uyari
                        if attack_ratio > 20:
                            st.markdown(
                                '<div class="attack-alert">'
                                f'⚠️ DIKKAT: Yuksek saldiri orani tespit edildi! '
                                f'({attack_ratio:.1f}% saldiri)'
                                '</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div class="normal-alert">'
                                f'✅ Ag trafigi buyuk oranda normal gorunuyor. '
                                f'({attack_ratio:.1f}% saldiri)'
                                '</div>',
                                unsafe_allow_html=True
                            )

                        # Grafikler
                        st.markdown("### 📊 Gorseller")
                        fig_dist = create_prediction_distribution_fig(predictions, probs)
                        st.pyplot(fig_dist)
                        plt.close()

                        # Gercek etiketler varsa - karsilastirma
                        if y_true is not None:
                            st.markdown("### 🎯 Model Performansi (Gercek Etiketlerle)")

                            acc = accuracy_score(y_true, predictions)
                            prec = precision_score(y_true, predictions, zero_division=0)
                            rec = recall_score(y_true, predictions, zero_division=0)
                            f1 = f1_score(y_true, predictions, zero_division=0)

                            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                            with mcol1:
                                st.metric("Dogruluk (Accuracy)", f"{acc * 100:.2f}%")
                            with mcol2:
                                st.metric("Kesinlik (Precision)", f"{prec * 100:.2f}%")
                            with mcol3:
                                st.metric("Duyarlilik (Recall)", f"{rec * 100:.2f}%")
                            with mcol4:
                                st.metric("F1-Score", f"{f1 * 100:.2f}%")

                            # Karisiklik matrisi
                            fig_cm = create_confusion_matrix_fig(
                                y_true, predictions, ["Normal", "Saldiri"]
                            )
                            st.pyplot(fig_cm)
                            plt.close()

                            # ROC egrisi
                            fig_roc, roc_auc_val = create_roc_curve_fig(
                                y_true, probs, selected_model
                            )
                            st.pyplot(fig_roc)
                            plt.close()

                            st.metric("ROC AUC", f"{roc_auc_val:.4f}")

                        # Detayli sonuc tablosu
                        with st.expander("📋 Detayli Tahmin Sonuclari", expanded=False):
                            result_df = pd.DataFrame({
                                "Ornek No": range(1, n_total + 1),
                                "Tahmin": ["SALDIRI" if p == 1 else "NORMAL" for p in predictions],
                                "Saldiri Olasiligi": [f"{p:.4f}" for p in probs],
                                "Guven": [f"{max(p, 1-p):.4f}" for p in probs],
                            })
                            if y_true is not None:
                                result_df["Gercek"] = ["SALDIRI" if y == 1 else "NORMAL" for y in y_true]
                                result_df["Dogru mu?"] = ["✅" if p == y else "❌" for p, y in zip(predictions, y_true)]

                            st.dataframe(result_df, use_container_width=True, height=400)

                            # CSV indirme
                            csv_data = result_df.to_csv(index=False)
                            st.download_button(
                                "📥 Sonuclari CSV Olarak Indir",
                                csv_data,
                                "tahmin_sonuclari.csv",
                                "text/csv"
                            )

# ============================================================
# Sekme 2: Model Egitimi
# ============================================================
with tab2:
    st.markdown("## 🏋️ Model Egitimi")
    st.markdown("CICIDS2017 veri seti ile modelleri egitin.")

    st.markdown('<div class="info-box">'
                '<strong>Not:</strong> Egitim islemi veri boyutuna gore birka dakika surebilir. '
                'Egitim sirasinda sayfa acik kalsin.'
                '</div>', unsafe_allow_html=True)

    col_train1, col_train2 = st.columns([1, 1])

    with col_train1:
        st.markdown("### 📋 Egitim Ayarlari")

        train_model_choice = st.selectbox(
            "Egitilecek Model:",
            ["Tumu (LSTM + RF + NB)", "Sadece LSTM", "Sadece Random Forest", "Sadece Naive Bayes"]
        )

        sample_size = st.number_input(
            "Orneklem Boyutu (her CSV icin):",
            min_value=1000,
            max_value=500000,
            value=10000,
            step=5000,
            help="Buyuk veri setlerinde hizli test icin orneklem alin"
        )

        classification_mode = st.radio(
            "Siniflandirma Modu:",
            ["Ikili (Normal / Saldiri)", "Coklu (Saldiri Turleri)"],
            help="Ikili: Normal vs Saldiri | Coklu: Her saldiri turu ayri sinif"
        )

    with col_train2:
        st.markdown("### 📁 Veri Seti")

        # CSV dosyalarini kontrol et
        csv_files = []
        if os.path.exists(DATA_DIR):
            csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

        if csv_files:
            st.success(f"✅ {len(csv_files)} CSV dosyasi bulundu:")
            for f in csv_files:
                fsize = os.path.getsize(os.path.join(DATA_DIR, f)) / (1024 * 1024)
                st.markdown(f"  - `{f}` ({fsize:.1f} MB)")
        else:
            st.warning(
                f"⚠️ `{DATA_DIR}` dizininde CSV dosyasi bulunamadi.\n\n"
                "CICIDS2017 veri setini indirip `data/` klasorune koyun veya "
                "asagidan sentetik veri olusturun."
            )

            if st.button("🔄 Sentetik Test Verisi Olustur (10.000 ornek)"):
                with st.spinner("Sentetik veri olusturuluyor..."):
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, "src/download_dataset.py",
                         "--generate-sample", "--n-samples", "10000"],
                        capture_output=True, text=True,
                        cwd=os.path.dirname(os.path.abspath(__file__))
                    )
                    if result.returncode == 0:
                        st.success("✅ Sentetik veri olusturuldu!")
                        st.rerun()
                    else:
                        st.error(f"Hata: {result.stderr}")

    st.markdown("---")

    # Egitim butonu
    can_train = len(csv_files) > 0

    if can_train:
        if st.button("🚀 Egitimi Baslat", type="primary", use_container_width=True):
            # Model parametresi
            model_map = {
                "Tumu (LSTM + RF + NB)": "all",
                "Sadece LSTM": "lstm",
                "Sadece Random Forest": "rf",
                "Sadece Naive Bayes": "nb"
            }
            model_arg = model_map[train_model_choice]
            is_multiclass = classification_mode.startswith("Coklu")

            # Egitim komutunu olustur
            cmd = [
                sys.executable, "src/train.py",
                "--model", model_arg,
                "--sample-size", str(sample_size),
            ]
            if is_multiclass:
                cmd.append("--multiclass")

            st.markdown("### 📺 Egitim Ciktisi")
            progress_placeholder = st.empty()
            output_placeholder = st.empty()

            with st.spinner("Model egitiliyor... Bu islem birka dakika surebilir."):
                import subprocess
                process = subprocess.run(
                    cmd,
                    capture_output=True, text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    timeout=600
                )

                if process.returncode == 0:
                    st.success("✅ Egitim basariyla tamamlandi!")
                    with st.expander("📋 Egitim Ciktisi", expanded=True):
                        st.code(process.stdout, language=None)
                    # Cache'i temizle
                    load_model_and_scaler.clear()
                    st.balloons()
                else:
                    st.error("❌ Egitim sirasinda hata olustu!")
                    st.code(process.stderr, language=None)
    else:
        st.info("Egitim baslatmak icin once `data/` klasorune CSV dosyalari ekleyin "
                "veya yukardaki 'Sentetik Test Verisi Olustur' butonunu kullanin.")

# ============================================================
# Sekme 3: Sonuclar ve Grafikler
# ============================================================
with tab3:
    st.markdown("## 📈 Egitim Sonuclari ve Grafikler")

    output_images = get_output_images()

    if output_images:
        # Degerlendirme raporu
        report_path = os.path.join(OUTPUTS_DIR, "evaluation_report.txt")
        if os.path.exists(report_path):
            with st.expander("📋 Degerlendirme Raporu", expanded=True):
                with open(report_path, "r") as f:
                    st.code(f.read(), language=None)

        st.markdown("---")
        st.markdown("### 📊 Grafikler")

        # Grafikleri kategorilere ayir
        comparison_imgs = {k: v for k, v in output_images.items()
                          if "comparison" in k or "model_" in k}
        cm_imgs = {k: v for k, v in output_images.items()
                   if "confusion" in k}
        roc_imgs = {k: v for k, v in output_images.items()
                    if "roc" in k}
        other_imgs = {k: v for k, v in output_images.items()
                      if k not in comparison_imgs and k not in cm_imgs
                      and k not in roc_imgs}

        # Karsilastirma grafikleri
        if comparison_imgs:
            st.markdown("#### 📊 Model Karsilastirmalari")
            for name, path in comparison_imgs.items():
                st.image(path, caption=name, use_container_width=True)

        # Karisiklik matrisleri
        if cm_imgs:
            st.markdown("#### 🎯 Karisiklik Matrisleri")
            cols_cm = st.columns(min(len(cm_imgs), 2))
            for i, (name, path) in enumerate(cm_imgs.items()):
                with cols_cm[i % 2]:
                    st.image(path, caption=name, use_container_width=True)

        # ROC egrileri
        if roc_imgs:
            st.markdown("#### 📈 ROC Egrileri")
            cols_roc = st.columns(min(len(roc_imgs), 2))
            for i, (name, path) in enumerate(roc_imgs.items()):
                with cols_roc[i % 2]:
                    st.image(path, caption=name, use_container_width=True)

        # Diger grafikler
        if other_imgs:
            st.markdown("#### 📉 Diger Grafikler")
            for name, path in other_imgs.items():
                st.image(path, caption=name, use_container_width=True)

    else:
        st.info("Henuz egitim yapilmadigi icin grafik yok. "
                "'Model Egitimi' sekmesinden egitim yapın.")

# ============================================================
# Sekme 4: Hakkinda
# ============================================================
with tab4:
    st.markdown("## ℹ️ Proje Hakkinda")

    st.markdown("""
    ### 🎯 Amac
    Bu proje, **CICIDS2017** veri seti kullanilarak ag trafigindeki siber saldirilari
    yapay zeka ile tespit etmeyi amaclamaktadir.

    ### 🧠 Kullanilan Modeller

    #### 1. LSTM (Long Short-Term Memory)
    - **Ana model** olarak kullanilmaktadir
    - 2 katmanli LSTM mimarisi (128 → 64 noron)
    - BatchNormalization ve Dropout katmanlari
    - Early Stopping ve Learning Rate Scheduler
    - Zaman serisi verilerindeki oruntuleri yakalama yetenegine sahiptir

    #### 2. Random Forest
    - 100 agacli topluluk ogrenmesi modeli
    - Hizli egitim ve tahmin suresi
    - Ozellik onemliligi analizi yapabilme

    #### 3. Naive Bayes (Gaussian)
    - Temel istatistiksel siniflandirici
    - En hizli egitim suresi
    - Basit ama etkili bir temel model

    ### 📊 Veri Seti: CICIDS2017
    - **Kaynak:** Canadian Institute for Cybersecurity
    - **Toplam:** 2.8+ milyon ag trafigi kaydi
    - **Saldiri Turleri:** DoS, DDoS, Brute Force, Web Attack, Infiltration, Bot, PortScan vb.
    - **Ozellikler:** 78+ ag trafigi ozelligi

    ### 📏 Degerlendirme Metrikleri
    - **Accuracy** (Dogruluk)
    - **Precision** (Kesinlik)
    - **Recall** (Duyarlilik)
    - **F1-Score**
    - **Confusion Matrix** (Karisiklik Matrisi)
    - **ROC Curve & AUC** (Alici Isletim Karakteristigi)

    ### 🛠️ Teknolojiler
    - Python 3.12
    - TensorFlow / Keras (LSTM)
    - Scikit-learn (RF, NB)
    - Streamlit (Web Arayuzu)
    - Pandas, NumPy, Matplotlib, Seaborn
    """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>"
        "Yapay Zeka Tabanli Siber Saldiri Tespit Sistemi | Bitirme Projesi"
        "</div>",
        unsafe_allow_html=True
    )
