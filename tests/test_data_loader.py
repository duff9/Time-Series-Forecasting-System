import forecasting_system.data_loader as dl


def test_read_data_file():
    data = dl.read_data_file('test_data.csv')
    assert (not data.empty)
    assert (len(data['Observation']) == 4)
