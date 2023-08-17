from forecasting_system import Model
from statsmodels.tsa.ar_model import AutoReg


class Autoregressive(Model.Model):
    """Use observations at previous time steps to predict observations at future time steps, univariate
    """

    def __init__(self, model_class, configuration):
        super().__init__(model_class, configuration)

    def train(self, training_data):
        self.trained_model = None

        if len(training_data['Observation'].values) > 0:

            model = AutoReg(
                training_data['Observation'].values,
                lags=self.configuration['lags']
            )

            self.trained_model = model.fit()

    def predict(self, forecast_data):
        start_step = len(self.training_data['Observation'].values)
        end_step = start_step + len(forecast_data.index.values) - 1
        forecast_data['Prediction'] = self.trained_model.predict(start=start_step, end=end_step)
        return forecast_data
        #super().predict(forecast_data)
