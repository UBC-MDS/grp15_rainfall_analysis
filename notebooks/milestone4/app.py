from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 1. Load your model here
model = joblib.load("model.joblib")

# 2. Define a prediction function
def return_prediction(inputs):
    # format input_data here so that you can pass it to model.predict()
    return model.predict([inputs])


# 3. Set up home page using basic html
@app.route("/")
def index():
    # feel free to customize this if you like
    return """
    <h1>Welcome to our rain prediction service</h1>
    To use this service, make a JSON post request to the /predict url with 5 climate model outputs.
    """


# 4. define a new route which will accept POST requests and return model predictions
@app.route("/predict", methods=["POST"])
def rainfall_prediction():
    content = request.json  # this extracts the JSON content we sent
    feats = content["data"]
    pred = return_prediction(feats)
    results = {"Input": feats, "Prediction": f"{pred[0]:.4f} mm/day"}
    return jsonify(results)