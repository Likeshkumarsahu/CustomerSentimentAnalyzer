from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Path to your saved model folder (MUST contain config.json)
MODEL_PATH = "sentiment_distilbert_best"

# Load pipeline correctly
classifier = pipeline(
    task="sentiment-analysis",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)

# Label mapping (FIX for LABEL_0 issue)
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Please enter some text to analyze.'})

    result = classifier(text)[0]

    sentiment = label_map.get(result['label'], result['label'])
    confidence = round(result['score'], 4)

    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)


