import forecasting_system.data_access_layer as dal
import forecasting_system.data_processor as dp
import datetime as dt


def test_halfhour_to_time():
    assert (dp.halfhour_to_time(1) == dt.time(0, 0, 0))
    assert (dp.halfhour_to_time(48) == dt.time(23, 30, 0))


def test_format_columns():
    data = dal.read_data_file('demanddata_2023.csv')
    data = dp.format_columns(data, 'ND', ['Observation', 'SETTLEMENT_PERIOD'])
    assert (not data.empty)
    assert (set(data.columns.to_list()) == set(['Observation', 'SETTLEMENT_PERIOD']))


def test_format_national_grid_data():
    data = dal.read_data_file('demanddata_2023.csv')
    data = dp.format_national_grid_data(data, 'ND', ['Observation', 'SETTLEMENT_PERIOD'])
    assert (not data.empty)
    assert (len(data['Observation']) > 1)
    assert (data['Observation'].dtype == "int64")
    assert (data.index.inferred_type == "datetime64")
