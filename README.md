# Healthcare Length of Stay Prediction Model

Accurately predicting patient **Length of Stay (LoS)** in healthcare institutions is critical for optimizing bed occupancy rates, streamlining staff planning, and reducing operational costs.

This project is an end-to-end machine learning solution that predicts a patient's hospital length of stay (in days) based on demographic information, chronic conditions, and laboratory results using the **Microsoft Hospital Length of Stay** dataset (100,000 patient records).

---

## Project Directory Structure

```text
hospital-length-of-stay-prediction/
├── data/                   # Kaggle veri setinin yerleştirileceği klasör
│   └── README.md
├── models/                 # Eğitilen modellerin (.joblib) kaydedildiği klasör
├── notebooks/              # Keşifsel Veri Analizi (EDA) ve görselleştirme notebook'ları
│   └── 01_eda_and_modeling.ipynb
├── reports/                # Model metrikleri ve görselleştirme çıktıları
│   └── figures/
├── src/                    # Modüler Python kaynak kodları
│   ├── __init__.py
│   ├── data_preprocessing.py  # Veri temizleme, özellik mühendisliği ve pipeline
│   ├── train.py               # Model eğitimi, karşılaştırma ve hiperparametre optimizasyonu
│   └── predict.py             # Yeni hasta verisi ile tahminleme modülü
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Kullanılan Teknolojiler ve Metodoloji

* **Veri Manipülasyonu & Analiz:** `Pandas`, `NumPy`
* **Görselleştirme:** `Matplotlib`, `Seaborn`
* **Makine Öğrenmesi:** `Scikit-Learn`, `XGBoost`
* **Özellik Mühendisliği (Feature Engineering):**
  * Kronik hastalık/komorbidite göstergelerinin birleştirilmesi (`total_comorbidities`, `has_comorbidity`)
  * Klinik oranların türetilmesi (`hema_neut_ratio`, `bmi_pulse_interaction`)
  * Tekrar yatış sayılarının (`rcount`) sayısal formata dönüştürülmesi
  * `ColumnTransformer` ve `Pipeline` mimarisi ile veri sızıntısını (data leakage) önleyen ölçeklendirme (`StandardScaler`) ve kodlama (`OneHotEncoder`)
* **Modelleme & Optimizasyon:** Ridge Regression, Random Forest ve XGBoost algoritmaları kıyaslanmış; `RandomizedSearchCV` ile hiperparametre optimizasyonu uygulanmıştır.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın ve Ortamı Kurun
```bash
git clone https://github.com/kullanici-adiniz/hospital-length-of-stay-prediction.git
cd hospital-length-of-stay-prediction

python -m venv venv
# Windows için:
venv\Scripts\activate
# Mac/Linux için:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Veri Setini Ekleyin
Kaggle üzerinden indirdiğiniz [Hospital Length of Stay Dataset Microsoft](https://www.kaggle.com/datasets/aayushchou/hospital-length-of-stay-dataset-microsoft) `.csv` dosyasını `data/` klasörünün içine yerleştirin.

### 3. Modeli Eğitin
```bash
python src/train.py
```
Bu komut çalıştırıldığında veriler işlenir, modeller eğitilir, hiperparametre optimizasyonu yapılır ve en başarılı model `models/best_los_model.joblib` olarak kaydedilir.

### 4. Örnek Tahmin Yapın
```bash
python src/predict.py
```

---

## 📊 Değerlendirme Metrikleri

Modeller regresyon problemi kapsamında aşağıdaki metriklerle değerlendirilmektedir:
* **RMSE (Root Mean Squared Error):** Büyük tahmin hatalarını cezalandıran kök ortalama kare hata.
* **MAE (Mean Absolute Error):** Tahminlerin gerçek kalış süresinden ortalama gün bazında sapması.
* **$R^2$ Skoru:** Modelin hedef değişkendeki varyansı açıklama oranı.
