import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

feature_names = [
    "No",
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude"
]

user_input = []

for feature in feature_names:
    while True:
        try:
            value = float(input(f"Введите значение для {feature}: "))
            user_input.append(value)
            break
        except ValueError:
            print("Ошибка! Введите число.")

X_new = pd.DataFrame([user_input], columns=feature_names)

predicted_price = model.predict(X_new)[0]

print(f"\nПредсказанная цена дома: {predicted_price:.2f}")