import importlib as il


def get_model_class(model_class):
    model_module = il.import_module('forecasting_system.models.' + model_class)
    model = getattr(model_module, model_class)
    return model


def create_model(model_class_name, configuration):
    model_class = get_model_class(model_class_name)
    model = model_class(model_class_name, configuration)
    return model


def train_model(model, observation_data):
    if model.trained_model is None:
        model.train(observation_data)


def predict_from_model(model, observation_data, forecast_data):
    train_model(model, observation_data)
    prediction = model.predict(forecast_data)
    return prediction
