import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Cargar dataset
df = pd.read_csv("Modelo3/modelo3_v1.csv")
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Dividir y entrenar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print("Precisión:", clf.score(X_test, y_test))

# Guardar modelo
with open("Modelo3/modelo3_v1.pkl", "wb") as f:
    pickle.dump(clf, f)
