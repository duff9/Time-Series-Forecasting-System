from forecasting_system import Model


class Persistence(Model.Model):
    """Uses the current observation as the prediction for the next time steps
    """

    def __init__(self, model_class, configuration):
        super().__init__(model_class, configuration)

    def train(self, observation_data):
        self.trained_model = None
        self.observation_data = observation_data

        if len(observation_data['Observation'].values) > 0:
            self.trained_model = observation_data['Observation'].values[-1]

    def predict(self, forecast_data):
        forecast_data['Prediction'] = self.trained_model
        return forecast_data
        #super().__predict(forecast_data)
