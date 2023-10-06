import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from datetime import date, datetime
from starlette.responses import JSONResponse
from joblib import load
from prophet.serialize import model_from_json


description = """

# Sales Forecasting API

Sales Forecast API helps you to do some awesome stuff.

## Usage

The API deploys 2 models:

* A predictive model that uses Catboost to predict the sales revenue for a given item in a specific store on a given date.
* A forecast model that uses Prophet to forecast the total sales revenue across all stores and items for the next 7 days.

## Endpoints

* ‘/’ (GET): Displaying a brief description of the project objectives, list of endpoints, expected input parameters and output format of the model, link to the Github repo related to this project
https://quiet-savannah-28493-e3545a2f3e86.herokuapp.com/

* ‘/health/’ (GET): Returning status code 200 with a string with a welcome message of your choice
https://quiet-savannah-28493-e3545a2f3e86.herokuapp.com/health/

* ‘/sales/national/’ (GET): Returning next 7 days sales volume forecast for an input date (date)
https://quiet-savannah-28493-e3545a2f3e86.herokuapp.com/sales/national?date=yyyy-mm-dd

* ‘/sales/stores/items/’ (GET): Returning predicted sales volume for an input item, store and date (item_id, store_id, date)
https://quiet-savannah-28493-e3545a2f3e86.herokuapp.com/sales/stores/item?item_id=item_id&store_id=store_id&date=yyyy-mm-dd

The input arguments need to be in the correct case and date format. The documentation for the application can be reviewed on:
https://quiet-savannah-28493-e3545a2f3e86.herokuapp.com/docs/


## Link to Github Repo

https://github.com/rushman95/timeseries-sales-forecast

"""

# Initialize app
app = FastAPI(
    title="Advanced ML AT2",
    description=description,
    summary="The Ultimate revenue forecasting solution",
    version="0.0.1")

# Load models
cat = load('../models/catboost.joblib')

with open('../models/prophet.json', 'r') as fin:
    prophet_pipe = model_from_json(fin.read())  # Save model


# Solution:
@app.get("/")
def read_root():
    return description

@app.get('/health', status_code=200)
def healthcheck():
    return 'This is a cool assignment! Thank you Anthony :)'

def format_features(
    item_id: str,
    store_id: str,
    d: date
    ):

    # Generating date and category features
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

    # Predicting
    pred = cat.predict(df)

    return JSONResponse(pred.tolist())

def format_date(
    d: date
    ):
    return {
        'date': d}


@app.get("/sales/national")
def forecast(date: date):

    # Generate a date range for the week and dataframe with these dates
    end_date = date + pd.DateOffset(days=6)
    date_range = pd.date_range(start=date, end=end_date)
    df = pd.DataFrame({'ds': date_range})

    # Forecasting on dataframe
    forecast = prophet_pipe.predict(df)

    result = forecast[['ds', 'yhat']]
    result['ds'] = pd.to_datetime(result['ds']).dt.date.astype(str)

    # Output the result as a dictionary
    result_dict = result.set_index('ds').to_dict()['yhat']

    return JSONResponse(result_dict)
