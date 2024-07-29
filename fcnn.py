import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report

# Dosya yollarını belirleme
file_paths = {
    'thumb': glob('D:/Ka YL/Tez/16_345/mutualData30/1bas/*.txt'),
    'index_finger': glob('D:/Ka YL/Tez/16_345/mutualData30/2isaret/*.txt'),
    'middle_finger': glob('D:/Ka YL/Tez/16_345/mutualData30/3orta/*.txt'),
    'ring_finger': glob('D:/Ka YL/Tez/16_345/mutualData30/4yuzuk/*.txt'),
    'little_finger': glob('D:/Ka YL/Tez/16_345/mutualData30/5serce/*.txt')
}

data = []
labels = []

# Verileri yükleme ve etiketleme
for finger, paths in file_paths.items():
    for file_path in paths:
        coeffs = np.loadtxt(file_path).reshape(-1)  # Veriyi düzleştir
        data.append(coeffs)
        labels.append(finger)

df = pd.DataFrame(data)
df['label'] = labels

X = df.drop('label', axis=1)
y = df['label']

# Etiketleri sayısal hale getirme
label_mapping = {label: idx for idx, label in enumerate(y.unique())}
y = y.map(label_mapping)

# Verileri ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verileri eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Etiketleri kategorik hale getirme
num_classes = len(label_mapping)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# FCNN modeli oluşturma
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Modeli derleme
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train_cat, epochs=50, batch_size=32, validation_data=(X_test, y_test_cat))

# Eğitim seti sonuçları
train_predictions = model.predict(X_train)
train_pred_classes = np.argmax(train_predictions, axis=1)
print("Eğitim Seti Doğruluk Oranı:", accuracy_score(y_train, train_pred_classes))
print("Eğitim Seti Sınıflandırma Raporu:\n", classification_report(y_train, train_pred_classes))

# Test seti sonuçları
test_predictions = model.predict(X_test)
test_pred_classes = np.argmax(test_predictions, axis=1)
print("Test Seti Doğruluk Oranı:", accuracy_score(y_test, test_pred_classes))
print("Test Seti Sınıflandırma Raporu:\n", classification_report(y_test, test_pred_classes))
