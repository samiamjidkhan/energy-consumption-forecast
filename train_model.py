import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

# Load the data
df = pd.read_csv('SF_hospital_load.csv')
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

# Create and fit the model
model = Prophet()
model.fit(df)

# Save the model using Prophet's serialization function
with open('serialized_model.json', 'w') as fout:
    fout.write(model_to_json(model))

print("Model training completed and saved.")
