# Healthcare Length of Stay Prediction Model

Accurately predicting patient **Length of Stay (LoS)** in healthcare institutions is critical for optimizing bed occupancy rates, streamlining staff planning, and reducing operational costs.

This project is an end-to-end machine learning solution that predicts a patient's hospital length of stay (in days) based on demographic information, chronic conditions, and laboratory results using the **Microsoft Hospital Length of Stay** dataset (100,000 patient records).

---

## Project Directory Structure

```text
hospital-length-of-stay-prediction/
├── data/                   # Directory where the Kaggle dataset should be placed
│   └── README.md
├── models/                 # Directory where trained models (.joblib) are saved
├── notebooks/              # Exploratory Data Analysis (EDA) and visualization notebooks
│   └── 01_eda_and_modeling.ipynb
├── reports/                # Model metrics and visualization outputs
│   └── figures/
├── src/                    # Modular Python source codes
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data cleaning, feature engineering, and pipeline
│   ├── train.py               # Model training, comparison, and hyperparameter tuning
│   └── predict.py             # Prediction module for new patient data
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Technologies and Methodology Used

* **Data Manipulation & Analysis:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning:** `Scikit-Learn`, `XGBoost`
* **Feature Engineering:**
  * Combining chronic disease/comorbidity indicators (`total_comorbidities`, `has_comorbidity`)
  * Deriving clinical ratios (`hema_neut_ratio`, `bmi_pulse_interaction`)
  * Converting readmission counts (`rcount`) into a numeric format
  * Preventing data leakage using `ColumnTransformer` and `Pipeline` rchitecture with (`StandardScaler`) and (`OneHotEncoder`)
* **Modeling & Optimization:** Compared Ridge Regression, Random Forest, and XGBoost algorithms; optimized hyperparameters using `RandomizedSearchCV`.

---

## 🚀 Installation and Usage

### 1. Clone the Repository and Set Up the Environment
```bash
git clone [https://github.com/falcon326/hospital-length-of-stay-prediction.git](https://github.com/falcon326/hospital-length-of-stay-prediction.git)
cd hospital-length-of-stay-prediction

python -m venv venv
# For Windows:
venv\Scripts\activate
# For Mac/Linux:
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
