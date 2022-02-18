# This file contains code to compute the filter function for
# a pulse sequence defined using rotation angles.

import numpy as np

def switching_function(pulse_rot, taus=None):
    """
    Function to generate the switching function list of given interpulse spacings

    Parameters
    ----------
    pulse_rot:
        pulse_rot : list
            sequence of pulse rotations - [a, b, c] is treated as [pi/a, pi/b, pi/c]
    taus : list, opt
        a list of tau spacings for eg [2, 3] corresponds to the sequence:
        I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I

    Returns
    -------
    switching_function : list
        Indicator function which flips between +1 and -1 when an rx pulse is applied.
        For eg: taus = [2, 3] should return:
        [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1]

    """
    if taus is None:
        taus = [5]*len(pulse_rot)

    # Convert pulse_rot list to angles
    pulse_rot = [2*np.pi / ele for ele in pulse_rot]

    # Compute length of phase_list
    length = 4 * sum(taus)

    # initialise phase list
    phase_list = [0]*length

    # add pulse rotations
    for idx, tau in enumerate(taus):

        # current position in the sequence
        pos =  4 * sum(taus[:idx])

        # add rx rotations
        phase_list[pos + tau] = pulse_rot[idx]
        phase_list[pos + 3 * tau ] = pulse_rot[idx]

    # generate switching function
    # print(phase_list)
    # switching_function = np.real(np.exp(1j * np.cumsum(phase_list)))
    f_yz = np.sin(np.cumsum(phase_list))
    f_zz = np.cos(np.cumsum(phase_list))

    # return switching_function
    return f_yz, f_zz

def filter_function(pulse_rot, time_scale=1, num_points=512,  taus=None):
    """
    Generate the filter function given a list of interpulse times

    Parameters
    ----------

    pulse_rot : list
    sequence of pulse rotations - [a, b, c] is treated as [pi/a, pi/b, pi/c]

    time_scale : float, optional
        time corresponding to one step. Defaults to 1.

    num_points : int, optional
        the number of points to be generated in the filter function. Defaults to 512.

    taus : list, opt
        a list of tau spacings for eg [2, 3] corresponds to the sequence:
        I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I

    Returns
    -------
    w : list
        frequencies over which the filter function is calculated

    FF : list
        Filter function corresponding to the frequencies in w


    """

    # Generate switching function
    # SF = switching_function(pulse_rot, taus=taus)
    f_yz, f_zz = switching_function(pulse_rot, taus=taus)

    # Generate frequencies
    w = np.arange(0, 2*num_points) * 2 * np.pi / (2*num_points*time_scale)

    # Compute the fourier transform of the switching function
    # F = np.fft.fft(SF,n=2*num_points) * time_scale
    F_yz = np.fft.fft(f_yz,n=2*num_points) * time_scale
    F_zz = np.fft.fft(f_zz,n=2*num_points) * time_scale

    # Compute the filter function
    # FF = np.abs(F)**2
    FF = np.abs(F_yz)**2 + np.abs(F_zz)**2

    # remove transients?
    FF = FF[:num_points]
    w = w[:num_points]

    return w, FF

def chi(pulse_rot, PSD, time_scale=1, taus=None):
    """
    Computes the decay constant given the inter-pulse spacing and power spectral density

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
    chi : float
        decay constant
    """


    w, FF = filter_function(pulse_rot, time_scale=time_scale, taus=taus)

    chi = np.trapz(FF * PSD / time_scale, w) / (2 * np.pi )
    return chi

def decay_probability(pulse_rot, PSD, time_scale=1, taus=None,zipped=False):
    """
    Computes the probability of measuring zero on average after applying the CPMG pulse

    Parameters
    ----------

    pulse_rot : list, opt
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

    if zipped:
        unzipped_object = zip(*pulse_rot)
        unzipped_list = list(unzipped_object)

        taus, pulse_rot  = unzipped_list
        taus = list(taus)
        pulse_rot = list(pulse_rot)

    p0 = 0.5 * (1 + np.exp(-chi(pulse_rot, PSD, time_scale=time_scale, taus=taus)) )
    return p0
