from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

data = pd.read_csv("real estate.csv")

X = data[[
    "No",
    "X1 transaction date",
    "X2 house age",
    "X3 distance to the nearest MRT station",
    "X4 number of convenience stores",
    "X5 latitude",
    "X6 longitude"
]]

y = data["Y house price of unit area"]

score = model.score(X, y)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            x0 = float(request.form["x0"])
            x1 = float(request.form["x1"])
            x2 = float(request.form["x2"])
            x3 = float(request.form["x3"])
            x4 = float(request.form["x4"])
            x5 = float(request.form["x5"])
            x6 = float(request.form["x6"])

            new_data = pd.DataFrame([[x0, x1, x2, x3, x4, x5, x6]], columns=X.columns)

            prediction = model.predict(new_data)[0]

        except:
            prediction = "Ошибка ввода"

    y_pred = model.predict(X)

    plt.figure()
    plt.scatter(y, y_pred)

    plt.plot([y.min(), y.max()], [y.min(), y.max()])

    plt.xlabel("Реальные цены")
    plt.ylabel("Предсказанные цены")
    plt.title("Реальные vs Предсказанные")

    if not os.path.exists("static"):
        os.makedirs("static")

    plt.savefig("static/plot.png")
    plt.close()

    table = data.head().to_html(classes="table", index=False)

    return render_template(
        "index.html",
        prediction=prediction,
        score=score,
        table=table
    )


if __name__ == "__main__":
    app.run(debug=True)