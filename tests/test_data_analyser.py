import forecasting_system.data_loader as dl
import forecasting_system.data_analyser as da


def test_calculate_correlation_coefficients():
    data = dl.read_data_file('test_data.csv')
    coeffs = da.calculate_correlation_coefficients(data)
    assert (not coeffs.empty)
    assert (set(coeffs.keys().to_list()) == {'Observation', 'Predictor'})
    assert (coeffs.shape == (2, 2))
    assert (coeffs.iloc[0, 0] == coeffs.iloc[1, 1] == 1)


def test_plot_correlation_heatmap():
    data = dl.read_data_file('test_data.csv')
    fig = da.plot_correlation_heatmap(data)
    assert (fig.axes[0].has_data())
    assert (fig.axes[0].get_title() == 'Correlation coefficients heatmap')


def test_plot_autocorrelation():
    data = dl.read_data_file('test_data.csv')
    fig = da.plot_autocorrelation(data['Observation'], 2)
    assert (fig.axes[0].has_data())
    assert (fig.axes[0].get_title() == 'Autocorrelation for: Observation')


def test_plot_partial_autocorrelation():
    data = dl.read_data_file('test_data.csv')
    fig = da.plot_partial_autocorrelation(data['Observation'], 1)
    assert (fig.axes[0].has_data())
    assert (fig.axes[0].get_title() == 'Partial autocorrelation for: Observation')


def test_calculate_principle_component_analysis():
    # TODO
    return
