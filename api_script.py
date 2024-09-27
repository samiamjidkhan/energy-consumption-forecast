from flask import Flask, request, jsonify
from marshmallow import Schema, fields, validate, ValidationError
import joblib
from prophet import Prophet

app = Flask(__name__)

class InputSchema(Schema):
    N = fields.Integer(required=True)

class OutputSchema(Schema):
    prediction = fields.Float()

input_schema = InputSchema()
output_schema = OutputSchema()

@app.route('/predict', methods=['GET'])
def predict():
    # Validate input
    try:
        data = input_schema.load(request.args)
    except ValidationError as err:
        return jsonify(err.messages), 400

    N = data['N']

    # Load the model
    try:
        model = joblib.load('prophet_model.joblib')
    except FileNotFoundError:
        return jsonify({"error": "Model file not found"}), 500

    # Make prediction
    future = model.make_future_dataframe(periods=N, freq='h')
    forecast = model.predict(future)

    # Get the last predicted value
    prediction = forecast['yhat'].iloc[-1]

    # Prepare and validate output
    output = output_schema.dump({"prediction": prediction})

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)