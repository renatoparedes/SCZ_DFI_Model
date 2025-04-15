import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import curve_fit
from skneuromsi.sweep import ProcessingStrategyABC
from scipy.signal import find_peaks

### FUNCTIONS


def compute_cost(model_data, exp_data):
    return np.sum(np.square(np.divide(exp_data - model_data, exp_data)))


def calculate_two_peaks_probability(visual_peaks_values):
    combinations = list(
        itertools.chain.from_iterable(
            itertools.combinations(visual_peaks_values, i + 2)
            for i in range(len(visual_peaks_values))
        )
    )

    probs_array = np.array([], dtype=np.float16)

    for i in combinations:
        probs_array = np.append(probs_array, np.array(i).prod())

    return probs_array.sum() / probs_array.size


def sig(x, a, b, c, d):
    """Compute sigmoidal value for the given delay.
    Args:
        x (1D np.array): Delay (ms).
        a (number): upper asymptote of the sigmoid.
        b (number): lower asymptote of the sigmoid.
        c (number): Central point of the sigmoid.
        d (number): Slope parameter of the sigmoid (slope = 1/d).

    Returns:
        sig (1D np.array): Vector with values for the given delay points.
    """
    return a + b / (1 + np.exp(-(x - c) / d))


## Sigmoid function fitting
def sigfit(x, y):
    """Fit the RT data to a sigmoidal function.

    Args:
        x (1D np.array): Delay (ms).
        y (1D np.array) : Values for the given delay points.

    Returns:
        a (number): upper asymptote of the sigmoid.
        b (number): lower asymptote of the sigmoid.
        c (number): Central point of the sigmoid.
        d (number): Slope parameter of the sigmoid (slope = 1/d).
    """

    # Obtains the upper and lower bounds
    a = np.max(y)
    b = np.min(y)

    # Defines starting points and boundaries for the fitting
    k_0 = (a - b) / (x[-1] - x[0])
    initial_slope = -(a - b) / (4 * k_0)
    # if initial_slope>=0: initial_slope=-0.0001
    middle_x = np.max(x) / 2
    init_guess = [a, b, middle_x, initial_slope]
    boundaries = ([0, 0, 36, float("-inf")], [100, 100, 204, 0])

    # Fits the data
    popt, _ = curve_fit(
        sig,
        x,
        y,
        p0=init_guess,
        method="trf",
        ftol=1e-8,
        xtol=1e-8,
        maxfev=100000,
        bounds=boundaries,
    )
    sigpar = np.asarray(popt)
    a = sigpar[0]
    b = sigpar[1]
    c = sigpar[2]
    d = sigpar[3]

    return a, b, c, d


def plot_res_per_soa(result_list, position=15):
    fig, axs = plt.subplots(3, 5, figsize=(16, 8), sharex=True, sharey=True)
    idx = -1
    for res in result_list:
        idx += 1
        row, col = idx // 5, idx % 5
        sub_plot = res.plot.linet(position=15, ax=axs[row, col])
        sub_plot.get_legend().remove()
        sub_plot.set_title("SOA " + str(int(res.run_params.soa)) + " ms")

    handles, labels = sub_plot.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", borderaxespad=0.1)
    plt.subplots_adjust(right=0.935)
    plt.show()


### CLASESS


class TwoFlashesProcessingStrategy(ProcessingStrategyABC):
    def map(self, result):
        max_pos = result.stats.dimmax().positions
        visual_activity = (
            result.get_modes(include="visual")
            .query(f"positions=={max_pos}")
            .visual.values
        )
        peaks, peaks_props = find_peaks(
            visual_activity,
            height=0.15,
            prominence=0.15,
            distance=36 / 0.01,
        )
        if len(peaks) < 2:
            p_two_flashes = 0
        else:
            p_two_flashes = calculate_two_peaks_probability(peaks_props["peak_heights"])
        del result._nddata
        return p_two_flashes * 100

    def reduce(self, results, **kwargs):
        return np.array(results, dtype=np.float16)


class TwoFlashesProcessingStrategy_Explore(ProcessingStrategyABC):
    def map(self, result):
        max_pos = result.stats.dimmax().positions
        visual_activity = (
            result.get_modes(include="visual")
            .query(f"positions=={max_pos}")
            .visual.values
        )
        peaks, peaks_props = find_peaks(
            visual_activity,
            height=0.15,
            prominence=0.15,
            distance=36 / 0.01,
        )
        if len(peaks) < 2:
            p_two_flashes = 0
        else:
            p_two_flashes = calculate_two_peaks_probability(peaks_props["peak_heights"])
        return result, p_two_flashes * 100

    def reduce(self, results, **kwargs):
        results_list = [res[0] for res in results]
        experiment_result = [res[1] for res in results]
        return results_list, experiment_result
