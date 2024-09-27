# Energy Consumption Forecasting Project

This project implements a time series forecasting model for energy consumption and provides an API endpoint to access predictions.

## Prerequisites

- Python
- pip (Python package installer)
- git

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/samiamjidkhan/energy-consumption-forecast.git
   cd energy-consumption-forecast
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

1. Ensure your data file `SF_hospital_load.csv` is in the project root directory.

2. Run the training script:
   ```
   python train_model.py
   ```

   This will create a `prophet_model.joblib` file in the project directory.

### Running the API

1. Start the Flask server:
   ```
   python api_script.py
   ```

2. To get a prediction, use a GET request to the `/predict` endpoint with the `N` parameter:
   ```
   curl "http://127.0.0.1:5000/predict?N=8"
   ```

   This will return a JSON response with the predicted energy consumption for 8 hours ahead.


## File Structure

- `train_model.py`: Script to train the Prophet model
- `api_script.py`: Flask API implementation
- `SF_hospital_load.csv`: Input data file
- `prophet_model.joblib`: Saved model file (generated after training)
- `requirements.txt`: List of Python dependencies

## Troubleshooting

- If you encounter a "Model file not found" error when running the API, make sure you've run the training script first.
- For any package installation issues, ensure you're using the correct version of Python and pip, and that your virtual environment is activated if you're using one.
