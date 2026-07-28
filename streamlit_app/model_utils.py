"""
model_utils.py
---------------
Memuat 3 model deep learning (ANN, CNN1D, LSTM) hasil training di notebook,
beserta scaler & urutan fitur, lalu menjalankan prediksi phishing/legitimate
untuk satu vektor fitur (hasil dari feature_extractor.py).
"""

import json
import os

import joblib
import numpy as np
import streamlit as st
import tensorflow as tf

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_FILES = {
    "ANN": "best_ANN.keras",
    "CNN1D": "best_CNN1D.keras",
    "LSTM": "best_LSTM.keras",
}

CLASS_NAMES = {0: "Legitimate (Aman)", 1: "Phishing"}


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Memuat scaler, urutan fitur, dan info model. Di-cache supaya hanya sekali baca disk."""
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    columns_path = os.path.join(MODEL_DIR, "feature_columns.json")
    info_path = os.path.join(MODEL_DIR, "model_info.json")

    if not os.path.exists(scaler_path) or not os.path.exists(columns_path):
        raise FileNotFoundError(
            "File scaler.pkl atau feature_columns.json tidak ditemukan di folder "
            f"'{MODEL_DIR}'. Pastikan notebook sudah dijalankan sampai selesai "
            "(bagian 'Menyimpan Artefak') sehingga artefak model tersedia."
        )

    scaler = joblib.load(scaler_path)
    with open(columns_path) as f:
        feature_columns = json.load(f)

    model_info = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            model_info = json.load(f)

    return scaler, feature_columns, model_info


@st.cache_resource(show_spinner=False)
def load_single_model(model_name: str):
    """Memuat satu model .keras dengan validasi warm-up, dan fallback yang jelas jika gagal."""
    path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File model '{path}' tidak ditemukan. Jalankan notebook training terlebih dahulu."
        )

    try:
        model = tf.keras.models.load_model(path, safe_mode=False)
    except Exception as e:
        raise RuntimeError(
            f"Gagal memuat model {model_name}: {e}. Kemungkinan ada mismatch versi "
            "TensorFlow/Keras antara environment training dan environment ini — "
            "cek `requirements.txt`."
        )

    # Warm-up: pastikan model benar-benar bisa menerima input dengan shape yang benar,
    # sebelum dipakai untuk prediksi sungguhan (mendeteksi masalah lebih awal & jelas).
    try:
        _, feature_columns, _ = load_artifacts()
        n_features = len(feature_columns)
        if model_name == "ANN":
            dummy = np.zeros((1, n_features), dtype=np.float32)
        else:
            dummy = np.zeros((1, n_features, 1), dtype=np.float32)
        model.predict(dummy, verbose=0)
    except Exception as e:
        raise RuntimeError(f"Model {model_name} termuat tetapi gagal saat warm-up: {e}")

    return model


def load_all_models(selected_models):
    """Memuat semua model yang dipilih. Mengembalikan (models_dict, errors_dict)."""
    models_dict, errors = {}, {}
    for name in selected_models:
        try:
            models_dict[name] = load_single_model(name)
        except Exception as e:
            errors[name] = str(e)
    return models_dict, errors


def vectorize_features(features: dict, feature_columns: list) -> np.ndarray:
    """Mengubah dict fitur (dari feature_extractor) menjadi array numpy sesuai urutan
    kolom yang dipakai saat training. Fitur yang tidak ada di-default 0 (netral)."""
    vec = np.array([[features.get(col, 0) for col in feature_columns]], dtype=np.float32)
    return vec


def predict_with_model(model, model_name: str, X_scaled: np.ndarray):
    """Menjalankan prediksi satu model. Mengembalikan dict hasil (label, confidence, prob)."""
    if model_name == "ANN":
        X_input = X_scaled
    else:
        X_input = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    prob_phishing = float(model.predict(X_input, verbose=0).ravel()[0])
    pred_label = 1 if prob_phishing >= 0.5 else 0

    return {
        "label": pred_label,
        "label_text": CLASS_NAMES[pred_label],
        "prob_phishing": prob_phishing * 100,
        "prob_legitimate": (1 - prob_phishing) * 100,
        "confidence": max(prob_phishing, 1 - prob_phishing) * 100,
    }


def predict_all(models_dict: dict, features: dict, feature_columns: list, scaler):
    """Menjalankan prediksi untuk seluruh model yang berhasil dimuat."""
    X_raw = vectorize_features(features, feature_columns)
    X_scaled = scaler.transform(X_raw)

    results = {}
    for name, model in models_dict.items():
        results[name] = predict_with_model(model, name, X_scaled)
    return results
