import os
import pandas as pd
import xgboost
from fastapi import FastAPI, HTTPException
from datetime import date, datetime
from starlette.responses import JSONResponse
from joblib import load
from prophet.serialize import model_to_json, model_from_json

app = FastAPI()

lr_pipe = load('../models/lr_pipeline.joblib')

with open('../models/serialized_model.json', 'r') as fin:
    prophet_pipe = model_from_json(fin.read())  # Save model

print(lr_pipe)

# Solution:
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Linear Regressor is all ready to go!'

def format_features(
    item_id: str,
    store_id: str,
    d: date
    ):
    return {
        'item_id': [item_id],
        'dept_id': [item_id.rsplit("_", 1)[0]],
        'cat_id': [item_id.split("_", 1)[0]],
        'store_id': [store_id],
        'state_id': [store_id.split('_')[0]],
        'year': [d.year],
        'month':[d.month],
        'day_of_week': [d.weekday()],
        'day_of_year': [(d - date(d.year, 1, 1)).days + 1]
    }

@app.get("/sales/stores/item")
def predict(
    item_id: str,
    store_id: str,
    date: date,
):
    features = format_features(
        item_id,
        store_id,
        date
        )

    df = pd.DataFrame(features)

    pred = lr_pipe.predict(df)

    return JSONResponse(pred.tolist())

def format_date(
    d: date
    ):
    return {
        'date': d}


@app.get("/sales/national")
def forecast(date: date):

    #date = format_date(date)
    print(date)
    # Generate a date range for the week
    end_date = date + pd.DateOffset(days=6)
    date_range = pd.date_range(start=date, end=end_date)

    df = pd.DataFrame({'ds': date_range})

    forecast = prophet_pipe.predict(df)

    result = forecast[['ds', 'yhat']]
    result['ds'] = pd.to_datetime(result['ds']).dt.date.astype(str)

    # Output the result as a dictionary
    result_dict = result.set_index('ds').to_dict()['yhat']

    return JSONResponse(result_dict)
