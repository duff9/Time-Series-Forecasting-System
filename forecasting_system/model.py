class Model:
    """Parent class that all model classes inherit from
    """

    trained_model = None
    training_data = None

    def __init__(self, model_class, configuration):
        self.model_class = model_class
        self.configuration = configuration

    def train(self, training_data):
        self.training_data = training_data
        return

    def predict(self, prediction_data):
        return prediction_data

    def reset(self):
        self.trained_model = None
        self.training_data = None

    def __repr__(self):
        # Custom display when using print on a model
        return "Model({}, {})".format(self.model_class, self.configuration)

    def __str__(self):
        # Custom display when using print on a model
        if self.trained_model is None:
            print_text = 'Model '
        else:
            print_text = 'Trained Model '

        return print_text + "({}, {}) {}".format(
            self.model_class, self.configuration, self.trained_model
        )
