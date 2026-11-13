import joblib
import pandas as pd
from data_preprocessing import engineer_features


def predict_patient_los(patient_data: dict, model_path: str = "models/best_los_model.joblib") -> float:
    """Tek bir hastanın verisini alarak tahmini yatış süresini (gün) döndürür."""
    model = joblib.load(model_path)
    df = pd.DataFrame([patient_data])
    df.columns = [c.strip().lower() for c in df.columns]
    df_processed = engineer_features(df)
    prediction = model.predict(df_processed)[0]
    return round(float(max(1.0, prediction)), 2)


if __name__ == "__main__":
    # Örnek bir hasta profili ile test
    sample_patient = {
        "rcount": "2",
        "gender": "F",
        "dialysisrenalendstage": 0,
        "asthma": 1,
        "irondef": 0,
        "pneum": 0,
        "substancedependence": 0,
        "psychologicaldisordermajor": 0,
        "depress": 0,
        "psychother": 0,
        "fibrosisandother": 0,
        "malnutrition": 0,
        "hemo": 0,
        "hematocrit": 11.5,
        "neutrophils": 14.2,
        "sodium": 138.0,
        "glucose": 145.0,
        "bloodureanitro": 18.0,
        "creatinine": 1.1,
        "bmi": 27.4,
        "pulse": 88,
        "respiration": 18.0,
        "secondarydiagnosisnonicd9": 2,
        "facid": "B"
    }

    try:
        estimated_days = predict_patient_los(sample_patient)
        print(f"Tahmini Hastanede Kalış Süresi: {estimated_days} gün")
    except FileNotFoundError:
        print("Önce 'python src/train.py' komutunu çalıştırarak modeli eğitmelisiniz.")
