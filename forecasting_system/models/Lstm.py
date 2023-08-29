from forecasting_system.Model import Model
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


class Lstm(Model):
    """A multivariate LSTM neural network model
    """

    x_mean = None
    x_std = None
    y_mean = None
    y_std = None

    def __init__(self, model_class, configuration):
        super().__init__(model_class, configuration)

    def train(self, training_data):
        train_x, train_y = super().pre_train_actions(training_data)

        # TODO split out training data to add validation data

        if self.configuration.get('normalise'):
            self.x_mean = train_x.mean()
            self.x_std = train_x.std()
            self.y_mean = train_y.mean()
            self.y_std = train_y.std()

            train_x = (train_x - self.x_mean) / self.x_std
            train_y = (train_y - self.y_mean) / self.y_std
            # validation_data = (validation_data - sample_mean) / sample_std

        # TODO differentiate between number of variables and sequence/observation length of a variable in input
        observations = train_x.shape[0]
        observation_length = 1
        variables = train_x.shape[1]

        # reshape to the input of RNN:
        # [number of observations/sequences, length of input observation/sequence, number of variables (features)]
        train_x = train_x.reshape((observations, observation_length, variables))

        # TODO make the layers part of the configuration? And the optimizer?
        model = Sequential([
            Input(shape=(observation_length, variables)),
            # keras handles the number of observations in the input_shape automatically
            LSTM(units=50, activation='relu'),
            Dense(units=1),
        ])

        model.summary()

        # TODO add default values for the configuration dictionary to use if not specified
        # TODO allow different optimisers
        model.compile(
            optimizer=Adam(learning_rate=self.configuration.get('Learning_rate')),
            loss=self.configuration.get('Loss_function')
        )

        auto_stop = EarlyStopping(
            monitor="loss",
            min_delta=0,
            patience=15,
            start_from_epoch=50
        )

        history = model.fit(
            train_x,
            train_y,
            epochs=self.configuration.get('Epochs'),
            callbacks=[auto_stop],
            verbose=0
        )

        if self.configuration.get('Plot_loss'):
            loss_values = history.history['loss']
            epochs = range(1, len(loss_values)+1)

            plt.plot(epochs, loss_values, label='Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        super().post_train_actions(model)

    def predict(self, forecast_data):

        predict_x = forecast_data[self.configuration.get('variables')].values

        if self.configuration('normalise'):
            predict_x = (predict_x - self.x_mean) / self.x_std

        predictions = predict_x.shape[0]
        observation_length = 1
        variables = predict_x.shape[1]
        predict_x = predict_x.reshape((predictions, observation_length, variables))

        predict_y = self.trained_model.predict(predict_x)

        if self.configuration('normalise'):
            predict_y = (predict_y * self.y_std) + self.y_mean

        forecast_data['Prediction'] = predict_y
        return forecast_data
        # #super().predict(forecast_data)
