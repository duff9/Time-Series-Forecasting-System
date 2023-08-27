import pandas as pd
import os
from zipfile import ZipFile
from keras.utils import get_file


def read_data_file(file_name):
    folder = 'data'
    file_path = os.path.join(folder, file_name)
    file = pd.read_csv(file_path, header=0, index_col=0)
    return file


def download_weather_data():
    uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    zip_path = get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
    zip_file = ZipFile(zip_path)
    zip_file.extractall('data')
    file_name = "jena_climate_2009_2016.csv"
    return read_data_file(file_name)
