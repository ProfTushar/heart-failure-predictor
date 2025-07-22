from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        float_inputs = [float(x) for x in request.form.values()]
        features = np.array(float_inputs).reshape(1, -1)
        prediction = model.predict(features)[0]
        result = "High Risk (Patient may die) 😢" if prediction == 1 else "Low Risk (Patient likely to survive) 😊"
    except:
        result = "Invalid input! Please enter correct values."

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
