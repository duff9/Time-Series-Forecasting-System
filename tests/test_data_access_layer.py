import forecasting_system.data_access_layer as dal
import datetime as dt


def test_read_data_file():
    data = dal.read_data_file('test_data.csv')
    assert(not data.empty)
    assert(len(data['Observation']) == 3)

def test_halfhour_to_time():
    assert(dal.halfhour_to_time(1) == dt.time(0, 0, 0))
    assert(dal.halfhour_to_time(48) == dt.time(23, 30, 0))
