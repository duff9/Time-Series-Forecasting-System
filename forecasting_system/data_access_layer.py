import pandas as pd
import os
import datetime as dt


def read_data_file(file_name):
    folder = 'data'
    file_path = os.path.join(folder, file_name)
    file = pd.read_csv(file_path, header=0, index_col=0)
    return file


def halfhour_to_time(halfhour):
    minutes = int((halfhour - 1) * 30)
    return dt.time(minutes // 60, minutes % 60, 0)
