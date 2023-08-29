class Model:
    """Parent class that all model classes inherit from
    """

    trained_model = None
    training_data = None

    def __init__(self, model_class, configuration):
        self.model_class = model_class
        self.configuration = configuration

    def pre_train_actions(self, training_data):
        self.trained_model = None
        self.training_data = training_data

        train_x = training_data[self.configuration.get('variables') or []].values
        train_y = training_data['Observation'].values

        if not len(training_data['Observation'].values) > 0:
            raise ValueError('''No observation data found,
                 observation data is required to train a model''')

        return train_x, train_y

    def post_train_actions(self, train_model):
        self.trained_model = train_model
        return

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
