import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots, correlation


def calculate_correlation_coefficients(dataframe):
    return dataframe.corr(method='pearson')


def plot_correlation_heatmap(dataframe):
    coeffs = calculate_correlation_coefficients(dataframe)
    correlation.plot_corr(
        coeffs,
        xnames=dataframe.columns.to_list(),
        ynames=dataframe.columns.to_list(),
        title="Correlation coefficients heatmap"
    )
    plt.show()


def plot_autocorrelation(series, lags=10):
    tsaplots.plot_acf(series, lags=lags, title="Autocorrelation for: " + series.name)
    plt.show()
