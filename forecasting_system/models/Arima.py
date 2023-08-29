from forecasting_system.Model import Model
from statsmodels.tsa.arima.model import ARIMA


class Arima(Model):
    """An autoregressive integrated moving average model, currently univariate
    """

    def __init__(self, model_class, configuration):
        super().__init__(model_class, configuration)

    def train(self, training_data):
        train_x, train_y = super().pre_train_actions(training_data)

        model = ARIMA(
            endog=train_y,
            exog=train_x,
            order=(
                self.configuration.get('AR_lags') or 0,
                self.configuration.get('differences') or 1,
                self.configuration.get('MA_lags') or 0
            )
        )

        trained_model = model.fit()
        super().post_train_actions(trained_model)

    def predict(self, forecast_data):
        predict_x = forecast_data[self.configuration.get('variables')].values

        start_step = len(self.training_data['Observation'].values)
        end_step = start_step + len(forecast_data.index.values) - 1
        forecast_data['Prediction'] = self.trained_model.predict(
            start=start_step,
            end=end_step,
            exog=predict_x
        )
        return forecast_data
