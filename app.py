import os
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Caminho e configuração
DATA_PATH = "mushroom.csv"
TARGET_COL = "class"

# ✅ Colunas usadas para treino e previsão
FEATURE_COLS = [
    "bruises", 
    "odor",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-color-above-ring",
    "spore-print-color",
    "population",
    "habitat"
]

st.set_page_config(page_title="Mushroom IA - Classificação", layout="wide")

st.title("🍄 Mushroom IA — Previsão: Comestível ou Venenoso")
st.markdown(
    "Este app usa **10 características principais** para prever se um cogumelo é comestível (e) ou venenoso (p)."
)

@st.cache_data
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo '{path}' não encontrado. Coloque o CSV na pasta do projeto.")
    df = pd.read_csv(path)
    return df

def is_boolean_like(series):
    unique = set(series.dropna().unique())
    bool_like_sets = [
        {True, False}, {"True", "False"}, {"true", "false"},
        {0, 1}, {"0", "1"}, {"t", "f"}, {"y", "n"}, {"yes", "no"}
    ]
    for s in bool_like_sets:
        if unique.issubset(s):
            return True
    return False

def preprocess(df, target_col=TARGET_COL):
    encoders = {}
    # ✅ Mantém apenas as colunas relevantes
    X = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
    y = df[target_col].copy()
    X_proc = pd.DataFrame(index=X.index)

    for col in X.columns:
        ser = X[col]
        if is_boolean_like(ser):
            mapping_vals = {
                True:1, False:0, 'True':1, 'False':0, 'true':1, 'false':0,
                't':1,'f':0,'y':1,'n':0,'yes':1,'no':0,'1':1,'0':0
            }
            X_proc[col] = ser.map(mapping_vals).fillna(0).astype(int)
        else:
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(ser.astype(str))
            encoders[col] = le

    target_le = LabelEncoder()
    y_enc = target_le.fit_transform(y.astype(str))

    return X_proc, y_enc, encoders, target_le

def build_sidebar_inputs(df, encoders):
    st.sidebar.header("🧠 Características para prever")
    inputs = {}
    X = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
    for col in X.columns:
        ser = X[col]
        if is_boolean_like(ser):
            most_common = ser.mode().iloc[0] if not ser.mode().empty else None
            default = False
            if isinstance(most_common, (int, float)):
                default = bool(most_common)
            else:
                default = str(most_common).lower() in ['true','t','y','yes','1']
            val = st.sidebar.checkbox(f"{col}", value=default)
            inputs[col] = int(val)
        else:
            uniques = list(ser.dropna().unique())
            uniques_str = [str(u) for u in uniques]
            default = uniques_str[0] if uniques_str else ""
            choice = st.sidebar.selectbox(f"{col}", options=uniques_str, index=0)
            if col in encoders:
                enc = encoders[col]
                try:
                    transformed = int(enc.transform([str(choice)])[0])
                except Exception:
                    transformed = 0
            else:
                transformed = 0
            inputs[col] = transformed
    return inputs

# Carregar dados
try:
    df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ✅ Filtrar apenas colunas desejadas + alvo
cols_needed = FEATURE_COLS + [TARGET_COL]
df = df[[c for c in cols_needed if c in df.columns]].copy()

st.subheader("📊 Amostra dos dados")
st.dataframe(df.head())

with st.expander("ℹ️ Informações das colunas usadas"):
    info = pd.DataFrame({
        'coluna': df.columns,
        'tipo': [str(t) for t in df.dtypes],
        'valores_únicos': [df[c].nunique() for c in df.columns]
    })
    st.dataframe(info)

if TARGET_COL in df.columns:
    fig = px.histogram(df, x=TARGET_COL, title="Distribuição da variável alvo (class)")
    st.plotly_chart(fig, use_container_width=True)

# Treinamento
st.subheader("⚙️ Treinamento do modelo")
st.markdown("O app irá treinar um **RandomForestClassifier** usando apenas as 10 colunas principais.")

X_proc, y_enc, encoders, target_le = preprocess(df, TARGET_COL)

test_size = st.slider("Tamanho do conjunto de teste (%)", 5, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(
    X_proc, y_enc, test_size=test_size/100.0, random_state=42
)

n_estimators = st.number_input("Número de árvores (n_estimators)", min_value=10, max_value=1000, value=100, step=10)
max_depth = st.number_input("Max depth (0 = None)", min_value=0, max_value=100, value=0, step=1)

if st.button("Treinar modelo agora"):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=(None if max_depth == 0 else int(max_depth)),
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"✅ Modelo treinado! Acurácia no teste: {acc:.4f}")
    st.text("Relatório de classificação:")
    st.text(classification_report(y_test, preds, target_names=target_le.classes_))
    st.session_state['model'] = rf
else:
    if 'model' in st.session_state:
        st.info("Modelo carregado da sessão anterior.")
    else:
        st.info("Clique em 'Treinar modelo agora' para treinar com os hiperparâmetros acima.")

# Previsão
if 'model' in st.session_state:
    model = st.session_state['model']
    user_inputs = build_sidebar_inputs(df, encoders)

    if st.button("🔮 Prever cogumelo"):
        feature_vec = [user_inputs[c] if c in user_inputs else 0 for c in FEATURE_COLS]
        feature_arr = np.array(feature_vec).reshape(1, -1)
        pred = model.predict(feature_arr)[0]
        proba = model.predict_proba(feature_arr)[0] if hasattr(model, "predict_proba") else None
        pred_label = target_le.inverse_transform([pred])[0]

        st.subheader("Resultado da previsão")
        if pred_label.lower().startswith('e'):
            st.success(f"🍽️ Previsto: COMESTÍVEL (label = {pred_label})")
        else:
            st.error(f"☠️ Previsto: VENENOSO (label = {pred_label})")

        if proba is not None:
            prob_df = pd.DataFrame({'classe': target_le.classes_, 'probabilidade': proba})
            st.table(prob_df)
else:
    st.info("Treine o modelo primeiro para habilitar previsões.")

st.markdown("""---
**Notas:**  
- Usa somente 10 colunas do dataset para treinamento e previsão.  
- Detecção automática de valores booleanos e codificação LabelEncoder.  
- Caso queira salvar o modelo em disco (`.joblib`), posso adicionar essa função.
""")

# Execução segura
if __name__ == "__main__":
    print("✅ Aplicação pronta! Rode com o comando abaixo no terminal:")
    print("\n    streamlit run app.py\n")
    print("Depois abra o link que aparecer (geralmente http://localhost:8501)")
