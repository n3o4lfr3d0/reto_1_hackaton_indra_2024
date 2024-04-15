import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Cargar el conjunto de datos
train_data = pd.read_csv('train_data.csv', delimiter=';')

# Dividir el conjunto de datos en características (X) y variable objetivo (y)
X = train_data.drop('abandono_6meses', axis=1)
y = train_data['abandono_6meses']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir las columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Definir las transformaciones para las columnas
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Utilizar Imputer para manejar los valores faltantes en las columnas numéricas
numeric_imputer = SimpleImputer(strategy='mean')

# Combinar las transformaciones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_imputer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Aplicar las transformaciones
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Inicializar el clasificador RandomForest
rf_classifier = RandomForestClassifier(random_state=42)

# Entrenar el clasificador
rf_classifier.fit(X_train_scaled, y_train)

# Predecir sobre el conjunto de prueba
y_pred = rf_classifier.predict(X_test_scaled)

# Calcular el F1 score
f1 = f1_score(y_test, y_pred)

print("F1 Score:", f1)

# Cargar el conjunto de datos de prueba
test_data = pd.read_csv('test_data.csv', delimiter=';')

# Aplicar las transformaciones al conjunto de datos de prueba
X_test_data = preprocessor.transform(test_data)

# Predecir sobre el conjunto de prueba
predictions = rf_classifier.predict(X_test_data)

# Crear un DataFrame con las predicciones
submission = pd.DataFrame({'ID': test_data['id_colaborador'], 'abandono_6meses': predictions})

# Guardar las predicciones en un archivo csv
submission.to_csv('predictions.csv', index=False)
