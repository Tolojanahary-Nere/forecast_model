# forecast.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Charger les données
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

# Visualisation
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Valeurs réelles')
plt.plot(y_pred, label='Prévisions', linestyle='--')
plt.legend()
plt.title("Prévision sur données temporelles")
plt.xlabel("Échantillons")
plt.ylabel("Valeurs")
plt.grid(True)
plt.tight_layout()
plt.show()
