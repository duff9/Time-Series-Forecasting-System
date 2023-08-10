import pandas as pd
from forecasting_system.data_access_layer import halfhour_to_time


def format_national_grid_data(data, obs_column):
    # convert date and period to datetime

    # TODO date and period in files are in local time not UTC, removing extra hour for now
    data.drop(data[data['SETTLEMENT_PERIOD'] > 48].index, inplace=True)

    data['Time'] = data['SETTLEMENT_PERIOD'].apply(
        lambda x: halfhour_to_time(x).strftime('%H:%M:%S'))
    data['Date_Time'] = pd.to_datetime(
        data.index + data['Time'],
        format='%Y-%m-%d%H:%M:%S')  # TODO format varies in different files
    data.reset_index(inplace=True)
    data.set_index('Date_Time', inplace=True)

    data.rename(columns={obs_column: "Observation"}, inplace=True)

    cols_to_keep = ['Observation', 'SETTLEMENT_PERIOD']

    data.drop(columns=[col for col in data.columns if col not in cols_to_keep], inplace=True)

    return data
