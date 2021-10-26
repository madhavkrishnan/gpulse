# This file contains different cost functions for the GA optimiser
from gpulse.ffn import decay_probability, chi
import numpy as np
from numpy.random import binomial


def new_fun():
    return 0


def ffn_cost(pulse_rot, PSD, time_scale=1, taus=None):
    """
    Computes the decay probability of a CPMG sequence individual based on the
    filter function

    Parameters
    ----------
    pulse_rot : list
        sequence of pulse rotations - [a, b, c] is treated as [pi/a, pi/b, pi/c]


    time_scale : float, optional
        time corresponding to one step. Defaults to 1.

    PSD : list
        power spectral density of the signal

    taus : list, opt
    a list of tau spacings for eg [2, 3] corresponds to the sequence:
    I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I

    Returns
    -------
    p0 : float
        probability of meausuring zero after the pulse sequence.

    """
    p0 = decay_probability(pulse_rot, PSD, time_scale=time_scale, taus=taus)

    return (p0,)

def ffn_noisy_cost(pulse_rot, PSD, time_scale=1, shots=100, taus=None):
    """
    Computes the decay probability of a CPMG sequence individual based on the
    filter function

    Parameters
    ----------
    pulse_rot : list
        sequence of pulse rotations - [a, b, c] is treated as [pi/a, pi/b, pi/c]


    time_scale : float, optional
        time corresponding to one step. Defaults to 1.

    PSD : list
        power spectral density of the signal

    taus : list, opt
    a list of tau spacings for eg [2, 3] corresponds to the sequence:
    I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I

    Returns
    -------
    tuple :
        outcome from binomial distrubution with parameter p computed from
        the filter function of the CPMG sequence.
    """

    p = ffn_cost(pulse_rot, PSD, time_scale=time_scale, taus=taus)

    return (float(binomial(shots, p)) / shots,)

def arb_cost(pulse, PSD, time_scale=1, shots=100, taus=None):
    """
    Computes the decay probability of a CPMG sequence individual based on the
    filter function

    Parameters
    ----------
    pulse_rot : list
        sequence of pulse rotations - [a, b, c] is treated as [pi/a, pi/b, pi/c]


    time_scale : float, optional
        time corresponding to one step. Defaults to 1.

    PSD : list
        power spectral density of the signal

    taus : list, opt
    a list of tau spacings for eg [2, 3] corresponds to the sequence:
    I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I

    Returns
    -------
    tuple :
        outcome from binomial distrubution with parameter p computed from
        the filter function of the CPMG sequence.
    """

    unzipped_object = zip(*pulse)
    unzipped_list = list(unzipped_object)

    taus, pulse_rot  = unzipped_list

    if len(taus) != len(pulse_rot):
        raise ValueError("Length of taus must equal length of rots")

    return ffn_noisy_cost(pulse_rot, PSD, time_scale=time_scale, shots=shots, taus=taus)



    # def ffn_noisy_cost_arb(pulse, PSD, time_scale=1, shots=100):
    #     """
    #     Computes the decay probability of a CPMG sequence individual based on the
    #     filter function.
    #
    #     Uses both interpulse spacing and pulse rotation.
    #
    #     Parameters
    #     ----------
    #     pulse : list
    #         list whose elements are two entry lists containting interpulse timing
    #         and rotation. E.g [[2,a], [3,b]] corresponds to the sequence:
    #         I-I-Rx(pi/a)-I-I-I-I-Rx(pi/a)-I-I + I-I-I-Rx(pi/b)-I-I-I-I-I-I-Rx(pi/b)-I-I-I
    #
    #     time_scale : float, optional
    #         time corresponding to one step. Defaults to 1.
    #
    #     PSD : list
    #         power spectral density of the signal
    #
    #
    #     Returns
    #     -------
    #     tuple :
    #         outcome from binomial distrubution with parameter p computed from
    #         the filter function of the CPMG sequence.
    #     """
    #
    #     pass
    #
    #     # unzipped_object = zip(*pulse)
    #     # unzipped_list = list(unzipped_object)
    #     #
    #     # taus, pulse_rot  = unzipped_list
    #     #
    #     # if len(taus) != len(pulse_rot):
    #     #     raise ValueError("Length of taus must equal length of rots")
    #     #
    #     # return 0
    #     # return ffn_noisy_cost(pulse_rot, PSD, time_scale=time_scale, shots=shots, taus=taus)

def cost_sig_noise(pulse, SIGNAL_PSD, NOISE_PSD, time_scale=1, shots=100):
    """
    Computes the difference in decay probability between noise and noise + singal
    of a CPMG sequence individual based on the filter function. I.e returns
    0.5 * e^{-X_noise}(1 - e^{-X_sig})

    Parameters
    ----------
    pulse : list
        sequence of pulse rotations and interpulse timing - [[t_1,a] , [t_2, b], [t_3,c]].
        Rotation angles will be [pi/a, pi/b, pi/c]


    time_scale : float, optional
        time corresponding to one step. Defaults to 1.

    SIGNAL_PSD : list
        power spectral density of the signal

    NOISE_PSD : list
        power spectral density of the Noise

    taus : list, opt
    a list of tau spacings for eg [2, 3] corresponds to the sequence:
    I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I

    Returns
    -------
    tuple :
        outcome from binomial distrubution with parameter p computed from
        the filter function of the CPMG sequence.
    """
    unzipped_object = zip(*pulse)
    unzipped_list = list(unzipped_object)

    taus, pulse_rot  = unzipped_list

    if len(taus) != len(pulse_rot):
        raise ValueError("Length of taus must equal length of rots")

    chi_sig = chi(pulse_rot, SIGNAL_PSD, time_scale=time_scale, taus=taus)
    chi_noise = chi(pulse_rot, NOISE_PSD, time_scale=time_scale, taus=taus)

    return 0.5 * np.exp(-chi_noise) * (1 - np.exp(-chi_sig)),
