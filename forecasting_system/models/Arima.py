from forecasting_system.Model import Model
from statsmodels.tsa.arima.model import ARIMA


class Arima(Model):
    """An autoregressive integrated moving average model, currently univariate
    """

    def __init__(self, model_class, configuration):
        super().__init__(model_class, configuration)

    def train(self, training_data):
        self.trained_model = None

        if len(training_data['Observation'].values) > 0:

            model = ARIMA(
                training_data['Observation'].values,
                order=(
                    self.configuration['AR_lags'] or 0,
                    self.configuration['differences'] or 1,
                    self.configuration['MA_lags'] or 0
                )
            )

            self.trained_model = model.fit()

    def predict(self, forecast_data):
        start_step = len(self.training_data['Observation'].values)
        end_step = start_step + len(forecast_data.index.values) - 1
        forecast_data['Prediction'] = self.trained_model.predict(start=start_step, end=end_step)
        return forecast_data
        #super().predict(forecast_data)
