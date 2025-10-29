import os
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

DATA_PATH = "mushroom.csv"
TARGET_COL = "class"

st.set_page_config(page_title="Mushroom IA - Classificação", layout="wide")

st.title("Mushroom IA — Previsão: Comestível ou Venenoso")
st.markdown("App que carrega um dataset de cogumelos (`mushroom.csv`), treina um modelo dentro do app e permite prever se um cogumelo é comestível (e) ou venenoso (p).")

@st.cache_data
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo '{path}' não encontrado. Coloque o CSV na pasta do projeto.")
    df = pd.read_csv(path)
    return df

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

def preprocess(df, target_col=TARGET_COL):
    encoders = {}
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    X_proc = pd.DataFrame(index=X.index)

    for col in X.columns:
        ser = X[col]
        if is_boolean_like(ser):
            try:
                numeric = pd.to_numeric(ser.dropna())
                if set(numeric.unique()).issubset({0,1}):
                    X_proc[col] = ser.astype(int)
                    continue
            except Exception:
                pass
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

# Load data
try:
    df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.subheader("Amostra dos dados (5 primeiras linhas)")
st.dataframe(df.head())

with st.expander("Visão geral das colunas e tipos"):
    info = pd.DataFrame({'coluna': df.columns, 'tipo': [str(t) for t in df.dtypes], 'valores_únicos': [df[c].nunique() for c in df.columns]})
    st.dataframe(info)

if TARGET_COL in df.columns:
    fig = px.histogram(df, x=TARGET_COL, title="Distribuição da variável alvo (class)")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Treinamento do modelo")
st.markdown("O app irá pré-processar automaticamente colunas char/bool e treinar um RandomForestClassifier.")
X_proc, y_enc, encoders, target_le = preprocess(df, TARGET_COL)

test_size = st.slider("Tamanho do conjunto de teste (%)", 5, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X_proc, y_enc, test_size=test_size/100.0, random_state=42)

n_estimators = st.number_input("Número de árvores (n_estimators)", min_value=10, max_value=1000, value=100, step=10)
max_depth = st.number_input("Max depth (0 = None)", min_value=0, max_value=100, value=0, step=1)

if st.button("Treinar modelo agora"):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=(None if max_depth==0 else int(max_depth)), random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Treinado! Acurácia no teste: {acc:.4f}")
    st.text("Relatório de classificação:")
    st.text(classification_report(y_test, preds, target_names=target_le.classes_))
    st.session_state['model'] = rf
else:
    if 'model' in st.session_state:
        st.info("Modelo carregado da sessão anterior.")
    else:
        st.info("Clique em 'Treinar modelo agora' para treinar com os hiperparâmetros acima.")

# === FORMULÁRIO CENTRAL DE PREDIÇÃO ===
if 'model' in st.session_state:
    model = st.session_state['model']

    st.subheader("Formulário de classificação de fungos")
    st.markdown("Responda todas as perguntas abaixo sobre o cogumelo e clique em 'Enviar para Análise'.")

    with st.form("mushroom_form"):
        q1 = st.radio("1) Escurece ao toque ou quando danificado (Bruises)?",
                      options=[('Sim/Yes','t'), ('Não/No','f')])
        q2 = st.radio("2) Odor?",
                      options=[('Cheiro Forte /Pungent','p'), ('Amêndoas/Almond','a'), 
                               ('Anis/Anise','l'), ('Nenhum/None','n'), ('Fétido/Foul','f'),
                               ('Creosoto/Creosote','c'), ('Peixe/Fishy','y'), ('Apimentado/Spicy','s'),
                               ('Mofo/Musty','m')])
        q3 = st.radio("3) Tamanho das lâminas (gill-size)?",
                      options=[('Estreitas/Narrow','n'), ('Largas/Broad','b')])
        q4 = st.radio("4) Cor das lâminas (gill-color)?",
                      options=[('Preta/Black','k'), ('Marrom/Brown','n'), ('Cinza/Gray','g'),
                               ('Rosa/Pink','p'), ('Branca/White','w'), ('Chocolate','h'),
                               ('Roxa/Purple','u'), ('Vermelha/Red','e'), ('Bege/Buff','b'),
                               ('Verde/Green','r')])
        q5 = st.radio("5) Formato do caule (stalk-shape):",
                      options=[('Alargado na base/Enlarging','e'), ('Afunilando/Tapering','t')])
        q6 = st.radio("6) Raiz do caule (stalk-root):",
                      options=[('Uniforme/Equal','e'), ('Em forma de clava/Club','c'),
                               ('Bulbosa/Bulbous','b'), ('Enraizada/Rooted','r')])
        q7 = st.radio("7) Cor do caule acima do anel (stalk-color-above-ring)?",
                      options=[('Branco/White','w'), ('Cinza/Gray','g'), ('Rosa/Pink','p'),
                               ('Marrom/Brown','n'), ('Bege/Buff','b'), ('Vermelho/Red','e'),
                               ('Laranja/Orange','o'), ('Canela/Cinnamon','c'), ('Amarelo/Yellow','y')])
        q8 = st.radio("8) Cor do esporo (spore-print-color)?",
                      options=[('Preto/Black','k'), ('Marrom/Brown','n'), ('Roxo/Purple','u'),
                               ('Chocolate','h'), ('Branco/White','w'), ('Verde/Green','r'),
                               ('Laranja/Oranje','o'), ('Amarelo/Yellow','y'), ('Bege/Buff','b')])
        q9 = st.radio("9) Como é a população (population)?",
                      options=[('Dispersa/Scattered','s'), ('Numerosa/Numerous','n'),
                               ('Abundante/Abundant','a'), ('Várias/Several','v'),
                               ('Solitária/Solitary','y'), ('Agrupada/Clustered','c')])
        q10 = st.radio("10) Habitat (habitat)?",
                      options=[('Urbano/Urban','u'), ('Gramados/Grasses','g'),
                               ('Prados/Meadows','m'), ('Florestas/Woods','d'),
                               ('Trilhas/Paths','p'), ('Terrenos baldios/Waste','w'),
                               ('Folhas/Leaves','l')])

        submitted = st.form_submit_button("Enviar para Análise")

        if submitted:
            user_inputs = {
                'bruises': 1 if q1[1]=='t' else 0,
                'odor': q2[1],
                'gill-size': q3[1],
                'gill-color': q4[1],
                'stalk-shape': q5[1],
                'stalk-root': q6[1],
                'stalk-color-above-ring': q7[1],
                'spore-print-color': q8[1],
                'population': q9[1],
                'habitat': q10[1]
            }

            feature_vec = []
            for col in X_proc.columns:
                val = user_inputs.get(col, 0)
                if col in encoders:
                    try:
                        val = encoders[col].transform([str(val)])[0]
                    except Exception:
                        val = 0
                feature_vec.append(val)
            feature_arr = np.array(feature_vec).reshape(1, -1)

            pred = model.predict(feature_arr)[0]
            pred_label = target_le.inverse_transform([pred])[0]

            if pred_label.lower().startswith('e'):
                st.success(f"🍽️ Previsto: COMESTÍVEL
