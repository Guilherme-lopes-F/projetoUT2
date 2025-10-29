import os
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# ---------------------------------------------
# CONFIGURAÇÕES INICIAIS
# ---------------------------------------------
DATA_PATH = "mushroom.csv"
TARGET_COL = "class"

st.set_page_config(page_title="🍄 Mushroom IA", layout="wide")

st.title("🍄 Mushroom IA — Classificação de Cogumelos")
st.markdown(
    "Treine um modelo de **Machine Learning** para prever se um cogumelo é "
    "**comestível (e)** ou **venenoso (p)** com base em todas as colunas do CSV."
)

# ---------------------------------------------
# FUNÇÃO PARA CARREGAR DADOS
# ---------------------------------------------
@st.cache_data
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"O arquivo '{path}' não foi encontrado.")
    df = pd.read_csv(path, sep="\t")  # CSV separado por tabulação
    return df

# ---------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------
def preprocess(df, target_col=TARGET_COL):
    encoders = {}
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    X_proc = pd.DataFrame(index=X.index)

    for col in X.columns:
        le = LabelEncoder()
        X_proc[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y.astype(str))
    return X_proc, y_enc, encoders, target_le

def build_sidebar_inputs(df, encoders):
    st.sidebar.header("🧩 Características do cogumelo")
    inputs = {}
    X = df.drop(columns=[TARGET_COL])
    for col in X.columns:
        uniques = sorted(list(df[col].dropna().unique()))
        choice = st.sidebar.selectbox(f"{col}", options=uniques, index=0)
        enc = encoders[col]
        inputs[col] = int(enc.transform([str(choice)])[0])
    return inputs

# ---------------------------------------------
# CARREGAR DATASET
# ---------------------------------------------
try:
    df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.subheader("📊 Amostra dos dados")
st.dataframe(df.head(10))

fig = px.histogram(df, x=TARGET_COL, title="Distribuição da variável alvo (class)")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------
# TREINAMENTO
# ---------------------------------------------
st.subheader("⚙️ Treinar Modelo")
X_proc, y_enc, encoders, target_le = preprocess(df, TARGET_COL)

test_size = st.slider("Tamanho do conjunto de teste (%)", 5, 50, 20)
n_estimators = st.number_input("Número de árvores (n_estimators)", 10, 500, 100, 10)
max_depth = st.number_input("Profundidade máxima (0 = None)", 0, 100, 0, 1)

if st.button("🚀 Treinar modelo agora"):
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y_enc, test_size=test_size/100, random_state=42, stratify=y_enc
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None if max_depth == 0 else int(max_depth),
        class_weight="balanced",
        random_state=42
    )

    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    st.success(f"✅ Modelo treinado com sucesso! Acurácia: {acc:.4f}")
    st.text("Relatório de classificação:")
    st.text(classification_report(y_test, preds, target_names=target_le.classes_, zero_division=0))

    st.session_state["model"] = rf
    st.session_state["encoders"] = encoders
    st.session_state["target_le"] = target_le
else:
    st.info("Clique no botão acima para treinar o modelo.")

# ---------------------------------------------
# PREVISÃO
# ---------------------------------------------
if "model" in st.session_state:
    st.subheader("🔮 Previsão de Novo Cogumelo")
    model = st.session_state["model"]
    encoders = st.session_state["encoders"]
    target_le = st.session_state["target_le"]

    user_inputs = build_sidebar_inputs(df, encoders)

    if st.button("🍄 Prever"):
        feature_vec = np.array([user_inputs[c] for c in df.drop(columns=[TARGET_COL]).columns]).reshape(1, -1)
        pred = model.predict(feature_vec)[0]
        proba = model.predict_proba(feature_vec)[0]

        pred_label = target_le.inverse_transform([pred])[0]
        if pred_label == "e":
            st.success(f"🍽️ Previsto: COMESTÍVEL ({pred_label})")
        else:
            st.error(f"☠️ Previsto: VENENOSO ({pred_label})")

        st.write("📊 Probabilidades:")
        st.dataframe(pd.DataFrame({
            "Classe": target_le.classes_,
            "Probabilidade": np.round(proba, 4)
        }))
else:
    st.info("Treine o modelo primeiro para habilitar previsões.")
