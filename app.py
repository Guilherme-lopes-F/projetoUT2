st.header("🔍 Previsão de cogumelo")

if "model" in st.session_state:
    with st.form("prediction_form"):
        user_data = {}

        # Exemplo para a primeira coluna
        st.markdown("**Classe do cogumelo (class)**")
        st.markdown("- p: poisonous / Venenoso")
        st.markdown("- e: edible / Comestível")
        user_data["class"] = st.text_input("Digite a opção correspondente (p/e)")

        # Exemplo para segunda coluna
        st.markdown("**Formato do chapéu (cap-shape)**")
        st.markdown("- x: Convexo / Convex")
        st.markdown("- b: Sino / Bell")
        st.markdown("- s: Afundado / Sunken")
        st.markdown("- f: Plano / Flat")
        st.markdown("- k: Nodoso / Knobbed")
        st.markdown("- c: Cônico / Conical")
        user_data["cap-shape"] = st.text_input("Digite a opção correspondente (x/b/s/f/k/c)")

        # ... repetir para todas as outras colunas ...

        submit = st.form_submit_button("Prever")

    if submit:
        df_user = pd.DataFrame([user_data])
        # Aplicar encoders
        for col, enc in st.session_state["encoders"].items():
            if col in df_user.columns and hasattr(enc, "transform"):
                df_user[col] = enc.transform(df_user[col].astype(str))

        prediction = st.session_state["model"].predict(df_user)[0]
        label = st.session_state["encoders"][TARGET_COL].inverse_transform([prediction])[0]
        st.success(f"🧠 Previsão: **{label}**")
else:
    st.warning("⚠️ Treine o modelo antes de tentar prever.")
