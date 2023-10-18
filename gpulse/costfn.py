# This file contains different cost functions for the GA optimiser
from gpulse.ffn import decay_probability, chi, filter_function
import numpy as np
from numpy.random import binomial
from gpulse.CPMG import CPMGCircuit
import qiskit as qk
from qiskit.providers.ibmq.managed import IBMQJobManager
from scipy.stats import binom


def new_fun():
    return 0


def ffn_cost(pulse_rot, PSD, time_scale=1, taus=None, ansatz = "CPMG"):
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

    ansatz : str, opt
        Expects either "CPMG" which is default or None. If CPMG for eg. tau
        spacings  [2, 3] corresponds to the sequence:
        I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I
        For anything else,
        I-I-Rx-I-I-I-Rx

    Returns
    -------
    p0 : float
        probability of meausuring zero after the pulse sequence.

    """
    p0 = decay_probability(pulse_rot, PSD, time_scale=time_scale, taus=taus, ansatz = ansatz)

    return (p0,)

def ffn_noisy_cost(pulse_rot, PSD, time_scale=1, shots=100, taus=None, ansatz = "CPMG"):
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

    ansatz : str, opt
        Expects either "CPMG" which is default or None. If CPMG for eg. tau
        spacings  [2, 3] corresponds to the sequence:
        I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I
        For anything else,
        I-I-Rx-I-I-I-Rx

    Returns
    -------
    tuple :
        outcome from binomial distrubution with parameter p computed from
        the filter function of the CPMG sequence.
    """

    p = ffn_cost(pulse_rot, PSD, time_scale=time_scale, taus=taus, ansatz = ansatz)

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

    ansatz : str, opt
        Expects either "CPMG" which is default or None. If CPMG for eg. tau
        spacings  [2, 3] corresponds to the sequence:
        I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I
        For anything else,
        I-I-Rx-I-I-I-Rx

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

def cost_sig_noise(pulse, SIGNAL_PSD, NOISE_PSD, time_scale=1, shots=100, ansatz="CPMG"):
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

    ansatz : str, opt
        Expects either "CPMG" which is default or None. If CPMG for eg. tau
        spacings  [2, 3] corresponds to the sequence:
        I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I
        For anything else,
        I-I-Rx-I-I-I-Rx

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

    chi_sig = chi(pulse_rot, SIGNAL_PSD, time_scale=time_scale, taus=taus, ansatz=ansatz)
    chi_noise = chi(pulse_rot, NOISE_PSD, time_scale=time_scale, taus=taus, ansatz=ansatz)

    return 0.5 * np.exp(-chi_noise) * (1 - np.exp(-chi_sig)),



def cost_sig_noise_shots(pulse, SIGNAL_PSD, NOISE_PSD, time_scale=1, shots=100, ansatz="CPMG"):
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

    ansatz : str, opt
        Expects either "CPMG" which is default or None. If CPMG for eg. tau
        spacings  [2, 3] corresponds to the sequence:
        I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I
        For anything else,
        I-I-Rx-I-I-I-Rx

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

    chi_sig = chi(pulse_rot, SIGNAL_PSD, time_scale=time_scale, taus=taus, ansatz=ansatz)
    chi_noise = chi(pulse_rot, NOISE_PSD, time_scale=time_scale, taus=taus, ansatz=ansatz)
    
    p_sig = 0.5*(1 + np.exp(-chi_noise - chi_sig))
    p_noise = 0.5*(1 + np.exp(-chi_noise))
    
    # p_sig_samp = np.mean(np.random.binomial(1, p_sig, shots))
    # p_noise_samp = np.mean(np.random.binomial(1, p_noise, shots))
    p_sig_samp = np.random.binomial(shots, p_sig)/shots
    p_noise_samp = np.random.binomial(shots, p_noise)/shots

    return (p_noise_samp - p_sig_samp),




def cost_sig_noise_sampling_error(pulse, SIGNAL_PSD, NOISE_PSD, time_scale=1, shots=100, ansatz="CPMG",
                                 sigma=0, num_trajs=1):
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

    ansatz : str, opt
        Expects either "CPMG" which is default or None. If CPMG for eg. tau
        spacings  [2, 3] corresponds to the sequence:
        I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I
        For anything else,
        I-I-Rx-I-I-I-Rx

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

    chi_sig = chi(pulse_rot, SIGNAL_PSD, time_scale=time_scale, taus=taus, ansatz=ansatz)
    chi_noise = chi(pulse_rot, NOISE_PSD, time_scale=time_scale, taus=taus, ansatz=ansatz)
    
    p_sig_perf = 0.5*(1 + np.exp(-chi_noise - chi_sig))
    p_noise_perf = 0.5*(1 + np.exp(-chi_noise))
    
    p_sig = 0
    p_noise = 0
    for _ in range(num_trajs):
        p_sig += p_sig_perf*(1 + np.random.normal(0, np.sqrt(sigma)))
        p_noise += p_noise_perf*(1 + np.random.normal(0, np.sqrt(sigma)))
    p_sig = p_sig/num_trajs
    p_noise = p_noise/num_trajs

    return (p_noise - p_sig),



def cost_sig_noise_coherent_error(pulse, SIGNAL_PSD, NOISE_PSD, time_scale=1, shots=100, error=0, ansatz="CPMG"):
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

    ansatz : str, opt
        Expects either "CPMG" which is default or None. If CPMG for eg. tau
        spacings  [2, 3] corresponds to the sequence:
        I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I
        For anything else,
        I-I-Rx-I-I-I-Rx

    Returns
    -------
    tuple :
        outcome from binomial distrubution with parameter p computed from
        the filter function of the CPMG sequence.
    """
    unzipped_object = zip(*pulse)
    unzipped_list = list(unzipped_object)

    taus, pulse_rot  = unzipped_list
    pulse_rot_array = np.array(pulse_rot)
    pulse_rot_array_error = pulse_rot_array/(1 + error)
    pulse_rot_err = tuple(pulse_rot_array_error)

    if len(taus) != len(pulse_rot):
        raise ValueError("Length of taus must equal length of rots")

    chi_sig = chi(pulse_rot_err, SIGNAL_PSD, time_scale=time_scale, taus=taus, ansatz = ansatz)
    chi_noise = chi(pulse_rot_err, NOISE_PSD, time_scale=time_scale, taus=taus, ansatz = ansatz)

    return 0.5 * np.exp(-chi_noise) * (1 - np.exp(-chi_sig)),




def cost_sig_noise_sampling_coherent_error(pulse, SIGNAL_PSD, NOISE_PSD, time_scale=1, shots=100, ansatz="CPMG",
                                 sigma=0, num_trajs=1, error=0, order=1):
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

    ansatz : str, opt
        Expects either "CPMG" which is default or None. If CPMG for eg. tau
        spacings  [2, 3] corresponds to the sequence:
        I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I
        For anything else,
        I-I-Rx-I-I-I-Rx

    Returns
    -------
    tuple :
        outcome from binomial distrubution with parameter p computed from
        the filter function of the CPMG sequence.
    """
    # print(sigma, num_trajs, error)
    unzipped_object = zip(*pulse)
    unzipped_list = list(unzipped_object)

    taus, pulse_rot  = unzipped_list
    pulse_rot_array = np.array(pulse_rot)
    pulse_rot_array_error = pulse_rot_array/(1 + error)
    pulse_rot_err = tuple(pulse_rot_array_error)

    if len(taus) != len(pulse_rot):
        raise ValueError("Length of taus must equal length of rots")


    chi_sig = chi(pulse_rot_err, SIGNAL_PSD, time_scale=time_scale, taus=taus, ansatz=ansatz)
    chi_noise = chi(pulse_rot_err, NOISE_PSD, time_scale=time_scale, taus=taus, ansatz=ansatz)
    
    p_sig_perf = 0.5*(1 + np.exp(-chi_noise - chi_sig))
    p_noise_perf = 0.5*(1 + np.exp(-chi_noise))
    
    # x = list(range(1,shots))
    # Bn = np.array([binom.pmf(i, shots, p_noise_perf) for i in x])
    # Bsn = np.array([binom.pmf(i, shots, p_sig_perf) for i in x])
    # Nth = find_N_threshold(Bn, Bsn)
    # error = np.sum(Bn[:Nth]) + np.sum(Bsn[Nth:])
    
    Nth = calc_N_threshold(p_noise_perf, p_sig_perf, shots)
    error = binom.cdf(Nth, shots, p_noise_perf) + (1 - binom.cdf(Nth, shots, p_sig_perf))
    return -error, 
    
#     p_sig = 0
#     p_noise = 0
#     for _ in range(num_trajs):
#         p_sig += np.max([0,np.min([p_sig_perf + np.random.normal(0, np.sqrt(sigma)),1])])
#         p_noise += np.max([0,np.min([p_noise_perf + np.random.normal(0, np.sqrt(sigma)),1])])
#     p_sig = p_sig/num_trajs
#     p_noise = p_noise/num_trajs

#     return (p_noise - p_sig),
    # p_sig_list = np.random.binomial(1, p_sig_perf, num_trajs)
    # p_noise_list = np.random.binomial(1, p_noise_perf, num_trajs)
    # dp = wasserstein_dist(np.sort(p_sig_list), np.sort(p_noise_list), p=order)
    # return dp,
    


def find_N_threshold(Bn, Bsn):
    # Bn (array) : binomial distribution for noise only
    # Bsn (array): binomial distribution for signal+noise
    Bn_max_idx = list(Bn).index(np.max(Bn))
    Bsn_max_idx = list(Bsn).index(np.max(Bsn))
    # print(Bn_max_idx, Bsn_max_idx)
    delta_B = np.abs(Bsn - Bn)
    if Bn_max_idx > Bsn_max_idx:
        B_thresh = np.min(delta_B[Bsn_max_idx:Bn_max_idx])
        N_thresh = list(delta_B).index(B_thresh)
    else:
        N_thresh = 0
    return N_thresh

def calc_N_threshold(p_n, p_sn, shots):
    if p_n > p_sn:
        Nth = int(np.round(shots*np.log10((1-p_sn)/(1-p_n))/np.log10((p_n*(1-p_sn))/((1-p_n)*p_sn)))) - 1
    else:
        Nth = 0
    return Nth

def wasserstein_dist(set1, set2, p=1):
    w12 = 0
    for i in range(len(set1)):
        w12 += np.abs(set1[i] - set2[i])**p
    return (1/len(set1)*w12)**(1/p)




def cost_sig_noise_qasm(pop, SIGNAL_CLASS, NOISE_CLASS, backend, flip_angle, time_scale=1, shots=100, num_sig_trajs=1, num_noise_trajs=1):
    # Constructs two circuits for each pulse sequence
    #   a) Circuit 1 is a signal-less circuit
    #   b) Circuit 2 contains the signal
    circuit_dict = {}
    for seq_idx, pulse_sequence in enumerate(pop):
        # Parse interpulse delays and pulse angles
        unzipped_object = zip(*pulse_sequence)
        unzipped_list = list(unzipped_object)
        taus, pulse_rot  = unzipped_list
        
        # Construct parameterized circuit
        cpmg_obj = CPMGCircuit(taus, pulse_rot, flip_angle)
        cpmg_circ = cpmg_obj.get_circuit()
        
        # grab parameters
        phis = cpmg_obj.phi_params
        
        # Construct signal-less circuit
        if num_noise_trajs == 0:
            phi_vals = np.zeros(len(phis))
            phi_dict = {phis[i]:phi_vals[i] for i in range(len(phis))}
            bound_cpmg = cpmg_circ.bind_parameters(phi_dict)
            circuit_dict['no-sig%d' % seq_idx] = bound_cpmg
        else:
            for traj_idx in range(num_noise_trajs):
                noise_traj_list = [NC.generate_noise(len(phis)) for NC in NOISE_CLASS]
                traj = np.sum(noise_traj_list, axis=0)
                phi_dict = {phis[i]:traj[i] for i in range(len(phis))}
                bound_cpmg = cpmg_circ.bind_parameters(phi_dict)
                circuit_dict['no-sig%d_traj%d' % (seq_idx, traj_idx)] = bound_cpmg
        
        # Construct signal circuit
        for traj_idx in range(num_sig_trajs):
            sig_traj = SIGNAL_CLASS.generate_noise(len(phis))
            noise_traj_list = [NC.generate_noise(len(phis)) for NC in NOISE_CLASS]
            noise_traj = np.sum(noise_traj_list, axis=0)
            traj = sig_traj + noise_traj
            phi_dict = {phis[i]:traj[i] for i in range(len(phis))}
            bound_cpmg = cpmg_circ.bind_parameters(phi_dict)
            circuit_dict['sig%d_traj%d' % (seq_idx, traj_idx)] = bound_cpmg
        
    # Run circuits
    if backend.name() == 'qasm_simulator':
        job = qk.execute(list(circuit_dict.values()), backend=backend, shots=shots, optimization_level=0)
        results = job.result()
    else:
        #job_manager = IBMQJobManager()
        # trans_circs = qk.transpile(list(circuit_dict.values()), backend=backend, optimization_level=0)
        # job_set = job_manager.run(trans_circs, backend=backend, name='var-sig-detect', shots=shots)
        #job_set = job_manager.run(list(circuit_dict.values()), backend=backend, name='var-sig-detect', shots=shots)
        #results = job_set.results()
        job = qk.execute(list(circuit_dict.values()), backend=backend, shots=shots, optimization_level=0)
        results = job.result()
    
    # Compile fitness
    no_sig_prob = []
    if num_noise_trajs == 0:
        for elem in range(len(pop)):
            circ = circuit_dict['no-sig%d' % elem]
            counts = results.get_counts(circ)
            zero_counts = counts.get('0',0)
            no_sig_prob.append(zero_counts/shots)
    else:
        for elem in range(len(pop)):
            traj_sum = 0
            for traj_idx in range(num_noise_trajs):
                circ = circuit_dict['no-sig%d_traj%d' % (elem, traj_idx)]
                counts = results.get_counts(circ)
                zero_counts = counts.get('0',0)
                traj_sum += zero_counts
            no_sig_prob.append(traj_sum/shots/num_sig_trajs)
    no_sig_prob = np.array(no_sig_prob)
    
    sig_prob = []
    for elem in range(len(pop)):
        traj_sum = 0
        for traj_idx in range(num_sig_trajs):
            circ = circuit_dict['sig%d_traj%d' % (elem, traj_idx)]
            counts = results.get_counts(circ)
            zero_counts = counts.get('0',0)
            traj_sum += zero_counts
        sig_prob.append(traj_sum/shots/num_sig_trajs)
    sig_prob = np.array(sig_prob)
    
    fitness = no_sig_prob - sig_prob
    return fitness




def cost_superres(pulse, wc, filter_null_band, time_scale=1, shots=100, ansatz="CPMG"):
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

    ansatz : str, opt
        Expects either "CPMG" which is default or None. If CPMG for eg. tau
        spacings  [2, 3] corresponds to the sequence:
        I-I-Rx-I-I-I-I-Rx-I-I + I-I-I-Rx-I-I-I-I-I-I-Rx-I-I-I
        For anything else,
        I-I-Rx-I-I-I-Rx

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
    
    w, FF = filter_function(pulse_rot, time_scale=time_scale, num_points=4112,  taus=taus, ansatz=ansatz)

    grad2FF = np.gradient(np.gradient(FF))
    # chi_sig = chi(pulse_rot, SIGNAL_PSD, time_scale=time_scale, taus=taus, ansatz=ansatz)
#     chi_noise = chi(pulse_rot, NOISE_PSD, time_scale=time_scale, taus=taus, ansatz=ansatz)
    # np.trapz(FF * PSD / time_scale, w) / (2 * np.pi )
    if len(filter_null_band) == 0:
        return -np.sum(FF[wc])/np.sum(FF) + np.sum(grad2FF[wc]),
    else:
        return -np.sum(FF[wc]) + np.sum(grad2FF[wc]) - np.sum(FF[filter_null_band]),