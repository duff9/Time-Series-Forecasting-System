from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA


def get_model_function(model_name):
    model_function = globals()['model_' + model_name]
    return model_function


def forecast_from_model(model_name, model_configuration, forecast_data, observation_data):
    model_function = get_model_function(model_name)
    forecast_data = model_function(forecast_data, observation_data, model_configuration)
    return forecast_data


def model_persistence(forecast_data, observation_data, configuration):
    # Uses current observation as the prediction for the next time step
    forecast_data['Prediction'] = forecast_data['Observation'].values[0]
    return forecast_data


def model_autoregressive(forecast_data, observation_data, configuration):
    # TODO split out training

    # Train model
    model = AutoReg(observation_data['Observation'].values, lags=configuration['lags'])
    model_fit = model.fit()

    # Forecast using trained model
    start_step = len(observation_data['Observation'].values)
    end_step = start_step + len(forecast_data.index.values)
    forecast_data['Prediction'] = model_fit.predict(start=start_step, end=end_step-1)

    return forecast_data


def model_arima(forecast_data, observation_data, configuration):
    # TODO split out training

    # Train model
    model = ARIMA(observation_data['Observation'].values, order=(configuration['lags'], 1, 0))  # order = (AR lags, number of differences, MA lags)
    model_fit = model.fit()

    # Forecast using trained model
    start_step = len(observation_data['Observation'].values)
    end_step = start_step + len(forecast_data.index.values)
    forecast_data['Prediction'] = model_fit.predict(start=start_step, end=end_step-1)

    return forecast_data
