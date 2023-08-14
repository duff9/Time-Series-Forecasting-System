"""Generalised interface for using different Models"""

import importlib as il
import pandas as pd
import datetime as dt
from forecasting_system.Prediction import Prediction


def get_model_class(model_class):
    """Obtain the model class constructor"""
    model_module = il.import_module('forecasting_system.models.' + model_class)
    model = getattr(model_module, model_class)
    return model


def create_model(model_class_name, configuration):
    """Create a model object using one of the classes in the "models" folder.
    All of the model classes inherit from the parent "Model" class.
    """
    model_class = get_model_class(model_class_name)
    model = model_class(model_class_name, configuration)
    return model


def train_model(model, observation_data, train_start_date=None, train_end_date=None):
    """Train a model using observation data.

    observation_data -- dataframe with datetime index and Observation column as dependent variable
    train_start_date -- date string, train from this date inclusive
    train_start_date -- date string, train to this date exclusive
    """
    model.reset()
    training_data = observation_data.copy()

    if train_start_date:
        training_data.drop(
            training_data[training_data.index < train_start_date].index,
            inplace=True
        )

    if train_end_date:
        training_data.drop(
            training_data[training_data.index >= train_end_date].index,
            inplace=True
        )

    model.training_data = training_data
    model.train(training_data)


def predict_from_model(
    model,
    prediction_data,
    forecast_start_date,
    steps,
    timestep=dt.timedelta(minutes=30)
):
    # TODO add error handling for when model is not trained

    # timestep is either dt.timedelta or dateoffset
    datetimes = pd.date_range(start=forecast_start_date, periods=steps, freq=timestep)
    prediction = pd.DataFrame({
        'Date_Time': datetimes,
        'Prediction': [None for t in range(len(datetimes))]
    })
    prediction.set_index('Date_Time', inplace=True)
    prediction = prediction.merge(prediction_data, how='left', left_index=True, right_index=True)

    prediction = Prediction(model.predict(prediction))

    return prediction
