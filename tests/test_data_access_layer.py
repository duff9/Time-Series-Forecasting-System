import forecasting_system.data_access_layer as dal


def test_read_data_file():
    data = dal.read_data_file('test_data.csv')
    assert(not data.empty)
    assert(len(data['Observation']) == 3)
