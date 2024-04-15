import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
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
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combinar las transformaciones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Definir el clasificador RandomForest
rf_classifier = RandomForestClassifier(random_state=42)

# Definir el pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', rf_classifier)])

# Definir los hiperparámetros para la búsqueda en cuadrícula
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

# Realizar la búsqueda en cuadrícula
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo y hacer predicciones
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calcular el F1 score
f1 = f1_score(y_test, y_pred)

print("Mejor modelo:", best_model)
print("F1 Score:", f1)

# Cargar el conjunto de datos de prueba
test_data = pd.read_csv('test_data.csv', delimiter=';')

# Predecir sobre el conjunto de prueba
predictions = best_model.predict(test_data)

# Crear un DataFrame con las predicciones
submission = pd.DataFrame({'ID': test_data['id_colaborador'], 'abandono_6meses': predictions})

# Guardar las predicciones en un archivo csv
submission.to_csv('predictions.csv', index=False)
