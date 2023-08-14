from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
import matplotlib.pyplot as plt


class Prediction:
    """Store and interact with predictions from models
    """

    metrics = None

    def __init__(self, prediction):
        self.prediction = prediction

    def calculate_error(self):
        if (not self.prediction['Observation'].empty):
            self.prediction['Error (P-O)'] = (
                self.prediction['Prediction'] - self.prediction['Observation']
            )

    def calculate_metrics(self):
        # TODO handle NaNs etc.
        observations = self.prediction['Observation'].values[1:]
        predictions = self.prediction['Prediction'].values[1:]
        mae = mean_absolute_error(observations, predictions)
        rmse = sqrt(mean_squared_error(observations, predictions))
        mape = mean_absolute_percentage_error(observations, predictions)
        self.metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        return self.metrics

    def plot(self, with_error=True):
        if with_error:
            self.calculate_error()
            self.prediction[['Prediction', 'Observation', 'Error (P-O)']].plot()
        else:
            self.prediction[['Prediction', 'Observation']].plot()
        plt.ylabel('Observation')
        plt.xlabel('Date_Time')
        plt.title('Prediction')
        plt.show()
