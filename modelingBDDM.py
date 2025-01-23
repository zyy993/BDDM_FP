# Import library yang diperlukan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

# 1. Memuat dataset
file_path = 'Indeks Standar Pencemar Udara di Provinsi DKI Jakarta 2021.csv'  # Pastikan file tersedia di direktori kerja
data = pd.read_csv(file_path)

# 2. Informasi Umum Dataset
print("Informasi Umum Dataset:")
print(data.info())

# 3. Preprocessing Data
# Isi nilai hilang di kolom 'pm25' dengan median
data['pm25'].fillna(data['pm25'].median(), inplace=True)

# Konversi kolom 'tanggal' ke tipe datetime (jika ada)
if 'tanggal' in data.columns:
    data['tanggal'] = pd.to_datetime(data['tanggal'], format='%m/%d/%Y')

# Encode kolom kategoris
categorical_columns = ['critical', 'categori', 'location']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 4. Definisikan fitur (X) dan target (y)
X = data[['pm10', 'pm25', 'so2', 'co', 'o3', 'no2', 'max', 'critical', 'location']]
y = data['categori']

# 5. Bagi data menjadi training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Bangun dan latih model Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Mendapatkan importance dari setiap fitur
feature_importance = model.feature_importances_

# 8. Menampilkan fitur berdasarkan tingkat importance
feature_names = X.columns
sorted_indices = np.argsort(feature_importance)[::-1]

print("Feature Importances:")
for idx in sorted_indices:
    print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")

# 9. Seleksi fitur berdasarkan threshold (contoh threshold: 0.05)
selected_features = [feature_names[idx] for idx in sorted_indices if feature_importance[idx] > 0.05]
print("\nSelected Features:", selected_features)

# 10. Dataset dengan fitur yang diseleksi
X_selected = X[selected_features]

# 11. Bagi ulang data dengan fitur yang diseleksi
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 12. Melatih ulang model dengan fitur yang diseleksi
model.fit(X_train, y_train)

# 13. Evaluasi model dengan data uji
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 14. Cetak hasil evaluasi
print("\nModel Accuracy with Selected Features:", accuracy)
print("\nClassification Report with Selected Features:\n", report)

# 15. Menyimpan model
joblib.dump(model, 'random_forest_model.pkl')
print("Model berhasil disimpan sebagai 'random_forest_model.pkl'")
