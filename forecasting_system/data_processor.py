import pandas as pd
import datetime as dt


def halfhour_to_time(halfhour):
    """Convert a halfhour period int to a datetime time"""
    minutes = int((halfhour - 1) * 30)
    return dt.time(minutes // 60, minutes % 60, 0)


def format_columns(data, obs_column, predictors):
    data.rename(columns={obs_column: "Observation"}, inplace=True)
    if predictors:
        cols_to_keep = ['Observation'] + predictors
        data.drop(columns=[col for col in data.columns if col not in cols_to_keep], inplace=True)
    return data


def format_national_grid_data(
    data,
    obs_column,
    predictors=['Observation', 'SETTLEMENT_PERIOD', 'EMBEDDED_SOLAR_GENERATION'],
    resample_rule=dt.timedelta(minutes=30)
):
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

    data = format_columns(data, obs_column, predictors)
    data = data.resample(resample_rule).first()
    return data


def format_jena_climate_data(data, obs_column, predictors, resample_rule=dt.timedelta(minutes=30)):
    data['Date_Time'] = pd.to_datetime(data.index, format='%d.%m.%Y %H:%M:%S')
    data.reset_index(inplace=True)
    data.set_index('Date_Time', inplace=True)
    data.drop(columns=['Date Time'], inplace=True)
    data = format_columns(data, obs_column, predictors)
    data = data.resample(resample_rule).mean()
    return data
