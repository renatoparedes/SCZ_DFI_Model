import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import curve_fit
from skneuromsi.sweep import ProcessingStrategyABC
from scipy.signal import find_peaks
from findpeaks import findpeaks
from matplotlib.ticker import ScalarFormatter

### FUNCTIONS


def adj_rmse(model_data, exp_data, k):
    sse = np.sum(np.square(exp_data - model_data))
    n = len(model_data)
    return np.sqrt(sse / (n - k))


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
    fig, axs = plt.subplots(3, 5, figsize=(16, 8), sharex=True, sharey=True, dpi=600)
    dashes_dict = {"auditory": "", "visual": "", "multi": (2, 2)}
    idx = -1
    for res in result_list:
        idx += 1
        row, col = idx // 5, idx % 5
        sub_plot = res.plot.linet(
            position=position,
            ax=axs[row, col],
            style="modes",
            dashes=dashes_dict,
            palette=["gray", "black", "gray"],
        )[0]
        sub_plot.get_legend().remove()
        sub_plot.set_title(
            "SOA " + str(int(res.run_parameters.auditory_soa)) + " ms",
            size=12,
            weight="bold",
        )
        sub_plot.set_ylabel("Neural activation", size=11, weight="bold")
        sub_plot.set_xlabel("Time (ms)", size=11, weight="bold")
        sub_plot.set_xlim(0, 60000)
        sub_plot.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        new_labels = [label.get_text()[:-2] for label in sub_plot.get_xticklabels()]
        if new_labels:
            new_labels[1] = "0"
        sub_plot.set_xticklabels(new_labels)
    handles, legend_labels = sub_plot.get_legend_handles_labels()
    new_legend_labels = ["Auditory", "Visual", "Multisensory"]
    fig.suptitle(None)
    fig.legend(handles, new_legend_labels, loc="center right", borderaxespad=0.75)
    plt.subplots_adjust(right=0.90)
    plt.show()


def plot_res_per_soa_small(result_list, position=15):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3), sharex=True, sharey=True, dpi=300)
    dashes_dict = {"auditory": "", "visual": "", "multi": (2, 2)}
    idx = -1
    for res in result_list[0::7]:
        idx += 1
        col = idx
        sub_plot = res.plot.linet(
            position=position,
            ax=axs[col],
            style="modes",
            dashes=dashes_dict,
        )[0]
        sub_plot.get_legend().remove()
        sub_plot.set_title(
            "SOA " + str(int(res.run_parameters.auditory_soa)) + " ms",
            size=12,
            weight="bold",
        )
        sub_plot.set_ylabel("Neural activation", size=11, weight="bold")
        sub_plot.set_xlabel("Time (ms)", size=11, weight="bold")
        sub_plot.set_xlim(0, 60000)
        sub_plot.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        new_labels = [label.get_text()[:-2] for label in sub_plot.get_xticklabels()]
        if new_labels:
            new_labels[1] = "0"
        sub_plot.set_xticklabels(new_labels)
    handles, legend_labels = sub_plot.get_legend_handles_labels()
    new_legend_labels = ["Auditory", "Visual", "Multisensory"]
    fig.legend(handles, new_legend_labels, loc="center right", borderaxespad=0.2)
    fig.suptitle(None)
    plt.subplots_adjust(wspace=0.1)
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
            p_two_flashes = (
                calculate_two_peaks_probability(peaks_props["peak_heights"]) * 100
            )
        del visual_activity, peaks, peaks_props, max_pos
        del result._nddata
        return p_two_flashes

    def reduce(self, results, **kwargs):
        return np.array(results, dtype=np.float16)


class TwoFlashesProcessingStrategy_Explore(ProcessingStrategyABC):
    def map(self, result):
        max_pos = result.stats.dimmax().positions
        ## Calculate two peak probability
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
            p_two_flashes = (
                calculate_two_peaks_probability(peaks_props["peak_heights"]) * 100
            )

        ## Calculate causes
        multi_activity = (
            result.get_modes(include="multi")
            .query(f"positions=={max_pos}")
            .multi.values
        )

        if (multi_activity > 0.80).any():
            multi_fp = findpeaks(method="topology", verbose=0, limit=0.8)
            multi_fp_results = multi_fp.fit(multi_activity)
            multi_peaks_df = multi_fp_results["df"].query(
                "peak==True & valley==False & y>0.80"
            )
            if multi_peaks_df["y"].size < 1:
                p_single_cause = 0
            elif multi_peaks_df["y"].size == 1:
                p_single_cause = multi_peaks_df["y"].values[0]
            else:
                p_single_cause = 1 - calculate_two_peaks_probability(
                    multi_peaks_df["y"].values
                )
        else:
            p_single_cause = 0

        return result, p_two_flashes, p_single_cause

    def reduce(self, results, **kwargs):
        results_list = [res[0] for res in results]
        experiment_result = [res[1] for res in results]
        causes_result = [res[2] for res in results]

        return results_list, experiment_result, causes_result


class BeepsProcessingStrategy(ProcessingStrategyABC):
    def map(self, result):
        max_pos = result.stats.dimmax().positions
        ## Calculate peaks
        visual_activity = (
            result.get_modes(include="visual")
            .query(f"positions=={max_pos}")
            .visual.values
        )
        if (visual_activity > 0.15).any():
            visual_fp = findpeaks(method="topology", verbose=0, limit=0.15)
            visual_fp_results = visual_fp.fit(visual_activity)
            visual_peaks_df = visual_fp_results["df"].query(
                "peak==True & valley==False & y>0.15"
            )
            n_flashes = visual_peaks_df["y"].size
        else:
            n_flashes = 0

        ## Calculate causes
        multi_activity = (
            result.get_modes(include="multi")
            .query(f"positions=={max_pos}")
            .multi.values
        )

        if (multi_activity > 0.80).any():
            multi_fp = findpeaks(method="topology", verbose=0, limit=0.8)
            multi_fp_results = multi_fp.fit(multi_activity)
            multi_peaks_df = multi_fp_results["df"].query(
                "peak==True & valley==False & y>0.80"
            )
            n_causes = multi_peaks_df["y"].size
        else:
            n_causes = 0

        del result._nddata
        return n_flashes, n_causes

    def reduce(self, results, **kwargs):
        n_flashes_list = [res[0] for res in results]
        n_causes_list = [res[1] for res in results]

        return n_flashes_list, n_causes_list
