import pandas as pd
from prophet import Prophet
import joblib

df = pd.read_csv('SF_hospital_load.csv')
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

model = Prophet()
model.fit(df)

joblib.dump(model, 'prophet_model.joblib')
