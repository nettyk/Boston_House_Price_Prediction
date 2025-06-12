# app.py

from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("model/boston_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input as float list
        features = [
            float(request.form["crim"]),
            float(request.form["zn"]),
            float(request.form["indus"]),
            float(request.form["chas"]),
            float(request.form["nox"]),
            float(request.form["rm"]),
            float(request.form["age"]),
            float(request.form["dis"]),
            float(request.form["rad"]),
            float(request.form["tax"]),
            float(request.form["ptratio"]),
            float(request.form["b"]),
            float(request.form["lstat"])
        ]

        # Scale input using trained scaler
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]

        return render_template("index.html", prediction_text=f"Predicted House Price: ${prediction * 1000:.2f}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
