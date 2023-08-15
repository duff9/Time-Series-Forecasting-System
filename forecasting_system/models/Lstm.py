from forecasting_system import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from keras.layers import LSTM
from keras.optimizers import Adam


class Lstm(Model.Model):
    """An LSTM neural network model
    """

    x_mean = None
    x_std = None
    y_mean = None
    y_std = None

    def __init__(self, model_class, configuration):
        super().__init__(model_class, configuration)

    def train(self, training_data):
        self.trained_model = None

        if len(training_data['Observation'].values) > 0:

            # Reshape data for LSTM model
            # TODO split out training data to add validation data?

            train_x = training_data[self.configuration['variables']].values
            train_y = training_data['Observation'].values

            if self.configuration['normalise']:
                self.x_mean = train_x.mean()
                self.x_std = train_x.std()
                self.y_mean = train_y.mean()
                self.y_std = train_y.std()

                train_x = (train_x - self.x_mean) / self.x_std
                train_y = (train_y - self.y_mean) / self.y_std
                # validation_data = (validation_data - sample_mean) / sample_std

            # TODO figure out how to differentiate between number of variables and sequence length of a variable in input
            observations = train_x.shape[0]
            observation_length = 1
            variables = train_x.shape[1]

            # reshape to the input of RNN:
            # [number of observations/sequences, length of input observation/sequence, number of variables (features)]
            train_x = train_x.reshape((observations, observation_length, variables))

            model = Sequential([
                Input(shape=(observation_length, variables)),
                # keras handles the number of observations in the input_shape
                LSTM(units=50, activation='relu'),
                Dense(units=1),
            ])
            model.summary()
            model.compile(optimizer=Adam(learning_rate=1.2), loss='mse')
            model.fit(train_x, train_y, epochs=200, verbose=0)
            self.trained_model = model

    def predict(self, forecast_data):

        predict_x = forecast_data[self.configuration['variables']].values

        if self.configuration['normalise']:
            predict_x = (predict_x - self.x_mean) / self.x_std

        predictions = predict_x.shape[0]
        observation_length = 1
        variables = predict_x.shape[1]
        predict_x = predict_x.reshape((predictions, observation_length, variables))

        predict_y = self.trained_model.predict(predict_x)

        if self.configuration['normalise']:
            predict_y = (predict_y * self.y_std) + self.y_mean

        forecast_data['Prediction'] = predict_y
        return forecast_data
        # #super().predict(forecast_data)
