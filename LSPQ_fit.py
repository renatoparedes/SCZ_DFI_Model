from utils import TwoFlashesProcessingStrategy
from skneuromsi.neural import Paredes2022
from skneuromsi.sweep import ParameterSweep

import numpy as np

from scipy.optimize import differential_evolution

lspq_data = np.array(
    [
        62.42548963,
        55.69572973,
        48.27845126,
        40.87320405,
        34.17511621,
        28.63657375,
        24.38672096,
        21.30902015,
        19.17234209,
        17.73209269,
        16.78048603,
        16.16001211,
        15.75893363,
        15.50112299,
        15.33600059,
    ],
    dtype=np.float16
)


def two_flashes_job(a_tau, v_tau, m_tau, cm_weight, fb_weight, ff_weight):

    model = Paredes2022(
        time_range=(0, 500),
        neurons=30,
        position_range=(0, 30),
        tau=(a_tau, v_tau, m_tau),
    )

    soas = np.array(
        [
            36.0,
            48.0,
            60.0,
            72.0,
            84.0,
            96.0,
            108.0,
            120.0,
            132.0,
            144.0,
            156.0,
            168.0,
            180.0,
            192.0,
            204.0,
        ]
        ,dtype=np.float16
    )

    sp = ParameterSweep(
        model=model,
        target="auditory_soa",
        repeat=1,
        n_jobs=1,
        range=soas,
        processing_strategy=TwoFlashesProcessingStrategy(),
    )

    res = sp.run(
        auditory_intensity=2.80,
        visual_intensity=1.75,
        auditory_stim_n=2,
        visual_stim_n=1,
        auditory_duration=7,
        visual_duration=12,
        noise=False,
        lateral_excitation=0.5,
        lateral_inhibition=0.4,
        cross_modal_weight=cm_weight,
        feedback_weight=fb_weight,
        feedforward_weight=ff_weight,
    )

    return res


def baseline_cost(theta):

    model_data = two_flashes_job(
        a_tau=theta[0],
        v_tau=theta[1],
        m_tau=theta[2],
        cm_weight=theta[3],
        fb_weight=theta[4],
        ff_weight=theta[5],
    )

    exp_data = lspq_data

    cost = np.sum(np.square(np.divide(exp_data - model_data, exp_data)))

    return cost


bounds = [(6, 10), (6, 30), (6, 120), (0.0001, 0.05), (0.0001, 0.25), (0.0001, 1)]

baseline_fit_res = differential_evolution(
    baseline_cost, bounds, disp=True, updating="deferred", workers=24, polish=False
)

print(baseline_fit_res)

np.save("LSPQ_fit_causal.npy", baseline_fit_res)
