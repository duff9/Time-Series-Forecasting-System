from forecasting_system.Model import Model
import forecasting_system.data_loader as dl


def test_init_model():
    model_class = 'test_model'
    config = {'test_config': 'yay'}
    model = Model(model_class, config)
    assert (type(model) == Model)
    assert (model.model_class == model_class)
    assert (model.configuration == config)


def test_train():
    model_class = 'test_model'
    config = {'test_config': 'yay'}
    model = Model(model_class, config)
    data = dl.read_data_file('test_data.csv')
    model.train(data)
    assert (type(model) == Model)
    assert (model.trained_model is None)
    assert (model.training_data.equals(data))


def test_predict():
    model_class = 'test_model'
    config = {'test_config': 'yay'}
    model = Model(model_class, config)
    data = dl.read_data_file('test_data.csv')
    model.train(data)
    f = model.predict(data)
    assert (f.equals(data))


def test_reset():
    model_class = 'test_model'
    config = {'test_config': 'yay'}
    model = Model(model_class, config)
    data = dl.read_data_file('test_data.csv')
    model.train(data)
    model.reset()
    assert (type(model) == Model)
    assert (model.trained_model is None)
    assert (model.training_data is None)
