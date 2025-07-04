# forecast.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("📈 Application de Prévision avec Régression Linéaire")
st.write("Ce modèle simple prévoit la prochaine valeur d'une série temporelle basée sur les précédentes.")

# Charger les données
try:
    df = pd.read_csv("data.csv", parse_dates=['date'])
    df['target'] = df['value'].shift(-1)
    df.dropna(inplace=True)

    # Séparation des variables
    X = df[['value']]
    y = df['target']

    # Séparer en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédiction
    y_pred = model.predict(X_test)

    # Affichage
    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    # Affichage du graphique
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

except FileNotFoundError:
    st.error("❌ Le fichier `data.csv` est introuvable. Assure-toi qu'il est bien dans le dépôt GitHub.")
except Exception as e:
    st.error(f"❌ Une erreur est survenue : {e}")
