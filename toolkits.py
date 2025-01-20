import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self, url, data_target):
        data = pd.read_csv(url)
        self.X = data[data_target]
        self.y = data['Produksi']
        self.scaler = StandardScaler()
        self.model = None

        x_train, x_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            random_state=42
        )

        # Scale data
        self.x_train_scaled = self.scaler.fit_transform(x_train)
        self.x_test_scaled = self.scaler.transform(x_test)
        self.y_train = y_train
        self.y_test = y_test

    def create_model(self, n_estimators, random_state):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        return self.model

    def train_model(self):
        if self.model is None:
            raise ValueError("Model belum dibuat. Gunakan create_model terlebih dahulu.")
        self.model.fit(self.x_train_scaled, self.y_train)

    def predict(self, new_data):
        if self.model is None:
            raise ValueError("Model belum dilatih. Gunakan train_model terlebih dahulu.")
        new_data_scaled = self.scaler.transform(np.array(new_data).reshape(1, -1))
        hasil_prediksi = self.model.predict(new_data_scaled)
        return float(hasil_prediksi)

# random_forest = RandomForestModel('dataset.csv', ['Tahun', 'Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata'])
# random_forest.create_model(n_estimators=100, random_state=42)
# random_forest.train_model()
# hasil = random_forest.predict([2023, 1500, 200, 75, 28])
# print("Hasil prediksi produksi:", hasil)