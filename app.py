from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)  # Debug print

    # Extract features and make prediction
    features = [data[key] for key in sorted(data.keys())]
    print("Features for prediction:", features)  # Debug print

    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
