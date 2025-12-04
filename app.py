from flask import Flask, render_template, request
import joblib

# Load trained model & vectorizer
model = joblib.load("Dataset/model/fake_news_model.pkl")
vectorizer = joblib.load("Dataset/model/tfidf_vectorizer.pkl")

# Initialize Flask
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form["news"]

    # Transform input text
    vector = vectorizer.transform([news_text])

    # Predict
    prediction = model.predict(vector)[0]

    result = "REAL NEWS ✅" if prediction == 1 else "FAKE NEWS ❌"

    return render_template("index.html", prediction=result, input_text=news_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
