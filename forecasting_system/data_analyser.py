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
    """The correlation between two values in the same time series"""
    tsaplots.plot_acf(series, lags=lags, title="Autocorrelation for: " + series.name)
    plt.show()


def plot_partial_autocorrelation(series, lags=10):
    """The correlation between two values excluding the correlations accounted for at smaller lags
    """
    tsaplots.plot_pacf(series, lags=lags, title="Partial autocorrelation for: " + series.name)
    plt.show()


def calculate_principle_component_analysis():
    # TODO
    return
