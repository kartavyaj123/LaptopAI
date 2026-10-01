from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained ML pipeline
with open("model/indian_laptop_price_pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)

# Load dataset for suggestions
df = pd.read_csv("data/laptops_raw.csv")


def clean_values(series):
    values = series.dropna().unique().tolist()
    return sorted([str(value) for value in values])


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predictor")
def predictor():
    return render_template(
        "predictor.html",
        prediction=None,
        error=None
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/get-suggestions")
def get_suggestions():
    return jsonify({
        "models": clean_values(df["model"]),
        "processors": clean_values(df["processor"]),
        "cores": clean_values(df["cores"]),
        "rams": clean_values(df["ram"]),
        "storages": clean_values(df["storage"]),
        "displays": clean_values(df["display"]),
        "graphics": clean_values(df["graphics"]),
        "operating_systems": clean_values(df["os"]),
        "warranties": clean_values(df["warranty"])
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        model = request.form.get("model", "").strip()
        rating_text = request.form.get("rating", "").strip()
        processor = request.form.get("processor", "").strip()
        cores = request.form.get("cores", "").strip()
        ram = request.form.get("ram", "").strip()
        storage = request.form.get("storage", "").strip()
        display = request.form.get("display", "").strip()
        graphics = request.form.get("graphics", "").strip()
        operating_system = request.form.get("os", "").strip()
        warranty = request.form.get("warranty", "").strip()

        rating = float(rating_text)

    except (ValueError, TypeError):
        return render_template(
            "predictor.html",
            prediction=None,
            error="Please enter a valid number for Rating."
        )

    # Validate rating
    if rating < 0 or rating > 100:
        return render_template(
            "predictor.html",
            prediction=None,
            error="Rating must be between 0 and 100."
        )

    # Validate required fields
    values = [
        model,
        processor,
        cores,
        ram,
        storage,
        display,
        graphics,
        operating_system,
        warranty
    ]

    if any(value == "" for value in values):
        return render_template(
            "predictor.html",
            prediction=None,
            error="Please fill in all laptop specifications."
        )

    # Create input DataFrame
    laptop_data = pd.DataFrame({
        "model": [model],
        "rating": [rating],
        "processor": [processor],
        "cores": [cores],
        "ram": [ram],
        "storage": [storage],
        "display": [display],
        "graphics": [graphics],
        "os": [operating_system],
        "warranty": [warranty]
    })

    # Make prediction
    prediction = pipeline.predict(laptop_data)[0]
    prediction = round(float(prediction), 2)

    return render_template(
        "predictor.html",
        prediction=prediction,
        error=None
    )


if __name__ == "__main__":
    app.run(debug=False)