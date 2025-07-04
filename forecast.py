# forecast.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("📈 Application de Prévision avec Régression Linéaire")
st.write("Chargez un fichier CSV contenant les colonnes `date` et `value`.")

uploaded_file = st.file_uploader("📂 Importez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['date'])
        df['target'] = df['value'].shift(-1)
        df.dropna(inplace=True)

        X = df[['value']]
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Aperçu des données")
        st.dataframe(df.head())

        st.subheader("📊 Résultats de la prévision")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test.values, label='Valeurs réelles')
        ax.plot(y_pred, label='Prévisions', linestyle='--')
        ax.legend()
        ax.set_title("Prévision sur données temporelles")
        ax.set_xlabel("Échantillons")
        ax.set_ylabel("Valeurs")
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Erreur lors du traitement du fichier : {e}")
else:
    st.info("📎 Veuillez importer un fichier CSV pour démarrer.")
