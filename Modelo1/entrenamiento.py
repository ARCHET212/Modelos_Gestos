import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Cargar dataset
df = pd.read_csv("Modelo1\modelo1_v1.csv")

# Separar X e y
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Entrenar RandomForest
clf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=25, 
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluar
y_pred = clf.predict(X_test)
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Guardar modelo
with open("Modelo1\modelo1_v1.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Modelo entrenado y guardado en 'modelo_lsg_estatico_dataset0.pkl'")
