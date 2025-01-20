
# Procedures For Use

### write and development by **kafnfie** 
model kit is a machine learning model creation tool built on scikit-learn py. This model is intended for beginners who want to create a simple machine learning model

# Model Random Forest Regressor
## understanding
Random forest regression is a supervised learning algorithm and bagging technique that uses an ensemble learning method for regression in machine learning. The trees in random forests run in parallel, meaning there is no interaction between these trees while building the trees.

## How to use

```
from toolkits import RandomForestModel as rfm

random_forest = rfm('dataset.csv', [p1, p2, p3, p4, p5])
random_forest.create_model(n_estimators=v1, random_state=v2)
random_forest.train_model()
result = random_forest.predict([p1, p2, p3, p4, p5])
print(f"predict: {result:.2f}")
```

## Example code

```
from toolkits import RandomForestModel as rfm

random_forest = rfm('dataset.csv', ['Tahun', 'Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata'])
random_forest.create_model(n_estimators=100, random_state=42)
random_forest.train_model()
hasil = random_forest.predict([2018, 408176.45, 2431.00, 80.00, 26.41])
print(f"Hasil prediksi produksi: {hasil:.2f}")
```


---
Donated at wallet BTC : bc1qcme5u6v8a4ss855jsvgae59z20f05sky494qpa  