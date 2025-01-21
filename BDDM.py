# Import library yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Memuat dataset
file_path = 'Indeks Standar Pencemar Udara di Provinsi DKI Jakarta 2021.csv'
data = pd.read_csv(file_path)

# 2. Preprocessing
# Isi nilai hilang di kolom 'pm25' dengan median
data['pm25'].fillna(data['pm25'].median(), inplace=True)

# Konversi kolom 'tanggal' ke tipe datetime
data['tanggal'] = pd.to_datetime(data['tanggal'], format='%m/%d/%Y')

# Encode kolom kategoris
categorical_columns = ['critical', 'categori', 'location']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 3. Definisikan fitur (X) dan target (y)
X = data[['pm10', 'pm25', 'so2', 'co', 'o3', 'no2', 'max', 'critical', 'location']]
y = data['categori']

# 4. Bagi data menjadi training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Bangun dan latih model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Cetak hasil evaluasi
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", report)
