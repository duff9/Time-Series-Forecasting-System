from forecasting_system.Model import Model
import forecasting_system.data_loader as dl
import numpy as np


def test_init_model():
    model_class = 'test_model'
    config = {'test_config': 'yay'}
    model = Model(model_class, config)
    assert (type(model) == Model)
    assert (model.model_class == model_class)
    assert (model.configuration == config)


def test_pre_train_actions():
    model_class = 'test_model'
    config = {'test_config': 'yay', 'variables': ['Predictor']}
    model = Model(model_class, config)
    data = dl.read_data_file('test_data.csv')
    train_x, train_y = model.pre_train_actions(data)
    assert (type(model) == Model)
    assert (model.trained_model is None)
    assert (model.training_data.equals(data))
    assert (type(train_y) == type(train_x) == np.ndarray)
    assert (len(train_y) == len(train_x))

    config = {'test_config': 'yay'}
    model = Model(model_class, config)
    train_x, train_y = model.pre_train_actions(data)
    assert (type(model) == Model)
    assert (model.trained_model is None)
    assert (model.training_data.equals(data))
    assert (type(train_y) == type(train_x) == np.ndarray)
    assert (len(train_y) == len(train_x))


def test_post_train_actions():
    model_class = 'test_model'
    config = {'test_config': 'yay'}
    model = Model(model_class, config)
    model.post_train_actions(model)
    assert (type(model) == Model)
    assert (model.trained_model is not None)


def test_reset():
    model_class = 'test_model'
    config = {'test_config': 'yay'}
    model = Model(model_class, config)
    data = dl.read_data_file('test_data.csv')
    model.pre_train_actions(data)
    model.reset()
    assert (type(model) == Model)
    assert (model.trained_model is None)
    assert (model.training_data is None)
