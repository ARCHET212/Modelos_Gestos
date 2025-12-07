import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle


ARCHIVO_CSV = "Modelo4/modelo4_v3.csv"
MODELO_PKL = "Modelo4/modelo4_v3.pkl"
EXPECTED_FEATURES = 83

try:
    # Cargar dataset
    df = pd.read_csv(ARCHIVO_CSV)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de datos en {ARCHIVO_CSV}. Por favor, ejecute primero el script de captura.")
    exit()


X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Verificar que las dimensiones sean correctas
if X.shape[1] != EXPECTED_FEATURES:
    print(f"ERROR CRÍTICO: Se esperaban {EXPECTED_FEATURES} características, pero se encontraron {X.shape[1]}.")
    print("El archivo CSV no es válido. Revise el script de captura o borre el CSV y capture de nuevo.")
    exit()

# División en train/test (80/20) con estratificación
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True,
    stratify=y, 
    random_state=42
)

# Entrenar RandomForest
clf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=25, 
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluación
y_pred = clf.predict(X_test)
print("--- Evaluación del Modelo 4 (v3: 83 features con Ángulos) ---")
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))
print(f"Precisión global: {clf.score(X_test, y_test)*100:.2f}%")

# Guardar modelo
with open(MODELO_PKL, "wb") as f:
    pickle.dump(clf, f)

print(f"\nModelo entrenado y guardado en '{MODELO_PKL}'")