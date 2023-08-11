import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
from forecasting_system.modeller import predict_from_model


def create_forecast(model, observation_data, start_date, steps, timestep):
    # timestep is either dt.timedelta or dateoffset

    datetimes = pd.date_range(start=start_date, periods=steps, freq=timestep)
    forecast = pd.DataFrame({
        'Date_Time': datetimes,
        'Prediction': [None for t in range(len(datetimes))]
    })
    forecast.set_index('Date_Time', inplace=True)
    forecast = forecast.merge(observation_data, how='left', left_index=True, right_index=True)

    observation_data.drop(observation_data[observation_data.index >= start_date].index, inplace=True)
    # print(observation_data.tail(), forecast.head())

    forecast = predict_from_model(model, observation_data, forecast)

    if (not forecast['Observation'].empty):
        forecast['Error'] = forecast['Prediction'] - forecast['Observation']
    error = calculate_error(forecast)

    return [forecast, error]


def calculate_error(forecast_data):
    observations = forecast_data['Observation'].values[1:]
    predictions = forecast_data['Prediction'].values[1:]
    mae = mean_absolute_error(observations, predictions)
    rmse = sqrt(mean_squared_error(observations, predictions))
    mape = mean_absolute_percentage_error(observations, predictions)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
