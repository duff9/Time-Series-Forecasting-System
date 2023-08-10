import forecasting_system.data_access_layer as dal
import forecasting_system.data_cleaner as dc


def test_format_national_grid_data():
    data = dal.read_data_file('demanddata_2023.csv')
    data = dc.format_national_grid_data(data, 'ND')
    assert(not data.empty)
    assert(len(data['Observation']) > 1)
    assert(data['Observation'].dtype == "int64")
    assert(data.index.inferred_type == "datetime64")
