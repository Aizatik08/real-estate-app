import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

file_path = "Real estate.csv"
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"CSV файл не найден: {file_path}")

X = data.drop("Y house price of unit area", axis=1)
y = data["Y house price of unit area"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

min_price = y_test.min()
max_price = y_test.max()
mean_price = y_test.mean()

pred_min = y_pred.min()
pred_max = y_pred.max()

print(f"Минимальная цена: {min_price}")
print(f"Максимальная цена: {max_price}")
print(f"Средняя цена: {mean_price:.6f}")
print(f"R²: {r2:.3f}\n")
print(f"Реальные цены: от {min_price:.2f} до {max_price:.2f}")
print(f"Предсказанные: от {pred_min:.2f} до {pred_max:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Предсказанные vs Реальные')
plt.plot([min_price, max_price], [min_price, max_price], color='red', linestyle='--', label='Идеальная линия')
plt.title("Linear Regression: Предсказанные vs Реальные цены")
plt.xlabel("Реальные цены")
plt.ylabel("Предсказанные цены")
plt.grid(True)
plt.legend()
plt.show()