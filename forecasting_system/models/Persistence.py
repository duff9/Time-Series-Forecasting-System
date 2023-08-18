from forecasting_system.Model import Model


class Persistence(Model):
    """Uses the current observation as the prediction for the next time steps
    """

    def __init__(self, model_class, configuration):
        super().__init__(model_class, configuration)

    def train(self, training_data):
        self.trained_model = None

        if len(training_data['Observation'].values) > 0:
            self.trained_model = training_data['Observation'].values[-1]

    def predict(self, forecast_data):
        forecast_data['Prediction'] = self.trained_model
        return forecast_data
        #super().__predict(forecast_data)
