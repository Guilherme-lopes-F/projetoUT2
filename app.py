import os
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Caminho e coluna-alvo
DATA_PATH = "mushroom.csv"
TARGET_COL = "class"

st.set_page_config(page_title="Mushroom IA - Form integrado", layout="wide")
st.title("🍄 Mushroom IA — Formulário integrado (previsão)")
st.markdown("Use o formulário abaixo para informar características do cogumelo. Treine o modelo (abaixo) antes de submeter o formulário para previsão.")

# --- Carregar dados ---
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        st.error(f"Arquivo '{path}' não encontrado. Coloque o CSV na pasta do projeto.")
        st.stop()
    return pd.read_csv(path)

# --- Detectar colunas booleanas ---
def is_boolean_like(series):
    unique = set(series.dropna().unique())
    bool_like_sets = [
        {True, False},
        {"True", "False"},
        {"true", "false"},
        {0, 1},
        {"0", "1"},
        {"t", "f"},
        {"y", "n"},
        {"yes", "no"}
    ]
    for s in bool_like_sets:
        if unique.issubset(s):
            return True
    return False

# --- Pré-processamento ---
def preprocess(df, target_col=TARGET_COL):
    encoders = {}
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    X_proc = pd.DataFrame(index=X.index)

    for col in X.columns:
        ser = X[col]
        if is_boolean_like(ser):
            mapping = {list(ser.dropna().unique())[0]: 0, list(ser.dropna().unique())[1]: 1}
            X_proc[col] = ser.map(mapping)
        elif ser.dtype == "object":
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(ser.astype(str))
            encoders[col] = le
        else:
            X_proc[col] = ser

    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)
    encoders[target_col] = le_y
    return X_proc, y_enc, encoders

# --- Carregar e treinar modelo ---
st.header("🔧 Treinamento do modelo")
if st.button("Treinar modelo"):
    df = load_data()
    X, y, encoders = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Modelo treinado com sucesso! 🎉 Acurácia: {acc:.2%}")

    st.session_state["model"] = model
    st.session_state["encoders"] = encoders
    st.session_state["X_cols"] = X.columns.tolist()

# --- Formulário de previsão ---
st.header("🔍 Previsão de cogumelo")
if "model" in st.session_state:
    with st.form("prediction_form"):
        user_data = {}
        for col in st.session_state["X_cols"]:
            user_data[col] = st.text_input(f"{col}")

        submit = st.form_submit_button("Prever")

    if submit:
        df_user = pd.DataFrame([user_data])
        # Converter entradas com encoders
        for col, enc in st.session_state["encoders"].items():
            if col in df_user.columns and hasattr(enc, "transform"):
                df_user[col] = enc.transform(df_user[col].astype(str))

        prediction = st.session_state["model"].predict(df_user)[0]
        label = st.session_state["encoders"][TARGET_COL].inverse_transform([prediction])[0]
        st.success(f"🧠 Previsão: **{label}**")
else:
    st.warning("⚠️ Treine o modelo antes de tentar prever.")
