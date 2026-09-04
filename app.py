import os
import pickle
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load vectorizer and model
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Inline HTML template with responsive CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis App</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f7f6;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }
        .container {
            background-color: #ffffff;
            padding: 30px 40px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            width: 100%;
        }
        h2 {
            margin-top: 0;
            color: #333333;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 6px;
            resize: vertical;
            font-size: 15px;
            box-sizing: border-box;
            margin-bottom: 20px;
        }
        button {
            width: 100%;
            background-color: #007bff;
            color: white;
            border: none;
            padding: 12px;
            font-size: 16px;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result-box {
            margin-top: 25px;
            padding: 15px;
            border-radius: 6px;
            background-color: #eef2f5;
        }
        .sentiment-badge {
            display: inline-block;
            padding: 6px 12px;
            font-weight: bold;
            border-radius: 4px;
            color: #fff;
        }
        .pos { background-color: #28a745; }
        .neg { background-color: #dc3545; }
        .neu { background-color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Review Sentiment Analyzer</h2>
        <form method="POST" action="/">
            <label for="review"><strong>Enter your review:</strong></label><br><br>
            <textarea id="review" name="review" placeholder="Type your review here..." required>{{ review }}</textarea>
            <button type="submit">Analyze Sentiment</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box">
            <h3>Analysis Result:</h3>
            <p><strong>Review:</strong> {{ review }}</p>
            <p><strong>Predicted Sentiment:</strong> 
                <span class="sentiment-badge {% if 'Pos' in prediction|string or prediction == 1 %}pos{% elif 'Neg' in prediction|string or prediction == 0 %}neg{% else %}neu{% endif %}">
                    {{ prediction }}
                </span>
            </p>
            {% if confidence %}
            <p><strong>Confidence:</strong> {{ confidence }}</p>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    review = ""
    
    if request.method == "POST":
        review = request.form.get("review", "")
        if review.strip():
            # Transform text input using vectorizer
            vect_text = vectorizer.transform([review])
            
            # Predict outcome
            pred = model.predict(vect_text)[0]
            prediction = pred
            
            # Extract probability if model supports predict_proba
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vect_text)[0]
                confidence = f"{max(probs) * 100:.2f}%"

    return render_template_string(HTML_TEMPLATE, review=review, prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
