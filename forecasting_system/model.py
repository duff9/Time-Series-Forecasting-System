class Model:
    """Parent class that all model classes inherit from
    """

    trained_model = None
    observation_data = None

    def __init__(self, model_class, configuration):
        self.model_class = model_class
        self.configuration = configuration

    def __train(self, observation_data):
        return

    def predict(self, forecast_data):
        return forecast_data

    def reset(self):
        self.trained_model = None
        self.observation_data = None
