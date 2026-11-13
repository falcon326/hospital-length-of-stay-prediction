import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def find_dataset_path(data_dir: str = "data") -> str:
    """Finds the first CSV file in the data directory."""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in the '{data_dir}' directory. Please add the Kaggle dataset here."
        )
    return csv_files[0]


def load_data(filepath: str = None) -> pd.DataFrame:
    """Loads the dataset and standardizes column names."""
    if filepath is None:
        filepath = find_dataset_path()
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies feature engineering for the Microsoft Hospital Length of Stay dataset."""
    df = df.copy()

    # Drop ID and date columns that could cause data leakage
    drop_cols = ["eid", "vdate", "discharged"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Comorbidity / chronic condition columns (standard Microsoft LoS columns)
    comorbidity_cols = [
        "dialysisrenalendstage", "asthma", "irondef", "pneum",
        "substancedependence", "psychologicaldisordermajor",
        "depress", "psychother", "fibrosisandother", "malnutrition", "hemo"
    ]
    existing_comorbidities = [c for c in comorbidity_cols if c in df.columns]

    if existing_comorbidities:
        # Ensure comorbidity columns are numeric
        for col in existing_comorbidities:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["total_comorbidities"] = df[existing_comorbidities].sum(axis=1)
        df["has_comorbidity"] = (df["total_comorbidities"] > 0).astype(int)

    # Clean readmission count ('5+' -> 5)
    if "rcount" in df.columns:
        df["rcount"] = df["rcount"].astype(str).str.replace("+", "", regex=False)
        df["rcount"] = pd.to_numeric(df["rcount"], errors="coerce").fillna(0)

    # Derive clinical indicators from vitals and lab values
    if {"hematocrit", "neutrophils"}.issubset(df.columns):
        df["hema_neut_ratio"] = df["hematocrit"] / (df["neutrophils"] + 1e-5)

    if {"bun", "creatinine"}.issubset(df.columns):
        df["bun_creatinine_ratio"] = df["bun"] / (df["creatinine"] + 1e-5)

    if {"bmi", "pulse"}.issubset(df.columns):
        df["bmi_pulse_interaction"] = df["bmi"] * df["pulse"]

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Builds the preprocessing pipeline for numerical and categorical features."""
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def prepare_train_test_data(filepath: str = None, target_col: str = "lengthofstay", test_size: float = 0.2, random_state: int = 42):
    """Loads data, applies feature engineering, and splits into train/test sets."""
    df = load_data(filepath)
    df = engineer_features(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}")

    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    preprocessor = build_preprocessor(X_train)
    return X_train, X_test, y_train, y_test, preprocessor
