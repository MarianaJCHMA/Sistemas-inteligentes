import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

url = "datosInstituo.cvs" # Cargar_datos
data = pd.read_csv(url)

print(data.head())

class_distribution = data['Class'].value_counts()
print("Distribución de las clases:")
print(class_distribution)

X = data.drop('Class', axis=1) 
y = data['Class']  

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

print("Tipo de problema: Clasificación binaria")

from sklearn.metrics import confusion_matrix

# matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Matriz de Confusión')
plt.xlabel('Predecir')
plt.ylabel('Actual')
plt.show()





