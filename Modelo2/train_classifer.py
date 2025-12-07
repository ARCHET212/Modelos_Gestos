import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



EXPECTED_FEATURES = 42

try:
    # Cargar los datos
    data_dict = pickle.load(open('Modelo2/modelo2_v1.pickle', 'rb'))
except FileNotFoundError:
    print("Error: El archivo no se encontró. Asegúrese de que esté en el directorio correcto.")
    exit()


cleaned_data = []
cleaned_labels = []

for i, features in enumerate(data_dict['data']):

    if len(features) == EXPECTED_FEATURES:
        cleaned_data.append(features)
        cleaned_labels.append(data_dict['labels'][i])


data = np.asarray(cleaned_data)
labels = np.asarray(cleaned_labels)


if data.size == 0:
    print("Error: Después de la limpieza, no quedan muestras de datos válidas (con 42 features).")
    print("Revise si su script de recolección guardó los datos correctamente.")
    exit()

print(f"Número de muestras válidas encontradas: {len(data)}")

x_train, x_test, y_train, y_test = train_test_split(
    data, 
    labels, 
    test_size=0.2, 
    shuffle=True, 
    stratify=labels 
)

# Inicializar y entrenar el modelo
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluación
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

# Mostrar el resultado
print('{}% de las muestras fueron clasificadas correctamente !'.format(round(score * 100, 2)))

# Guardar el modelo entrenado
f = open('Modelo2/model_v1.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

print("El modelo ha sido entrenado y guardado")