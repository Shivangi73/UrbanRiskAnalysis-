import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("disease_data.csv")

# Features & target
X = data[['Temperature', 'Humidity', 'AQI', 'Rainfall', 'Population_Density']]
y = data['Risk_Level']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model
model = pickle.load(open("disease_model.pkl", "rb"))

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("Labels order: Low, Medium, High")
print(cm)

# ---------------- CLASSIFICATION REPORT ----------------
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)