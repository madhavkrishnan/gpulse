# This file contains different cost functions for the GA optimiser
from gpulse.ffn import decay_probability, chi
import numpy as np
from numpy.random import binomial
from gpulse.CPMG import CPMGCircuit
import qiskit as qk
from qiskit.providers.ibmq.managed import IBMQJobManager


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
    pulse_rot_array_error = pulse_rot_array*(1 + error)
    pulse_rot_err = tuple(pulse_rot_array_error)

    if len(taus) != len(pulse_rot):
        raise ValueError("Length of taus must equal length of rots")

    chi_sig = chi(pulse_rot_err, SIGNAL_PSD, time_scale=time_scale, taus=taus, ansatz = anstaz)
    chi_noise = chi(pulse_rot_err, NOISE_PSD, time_scale=time_scale, taus=taus, ansatz = ansatz)

    return 0.5 * np.exp(-chi_noise) * (1 - np.exp(-chi_sig)),



def cost_sig_noise_qasm(pop, SIGNAL_CLASS, NOISE_CLASS, backend, time_scale=1, shots=100, num_sig_trajs=1, num_noise_trajs=1):
    # Constructs two circuits for each pulse sequence
    #   a) Circuit 1 is a signal-less circuit
    #   b) Circuit 2 contains the signal
    circuit_dict = {}
    for idx, pulse_sequence in enumerate(pop):
        # Parse interpulse delays and pulse angles
        unzipped_object = zip(*pulse_sequence)
        unzipped_list = list(unzipped_object)
        taus, pulse_rot  = unzipped_list

        # Construct parameterized circuit
        measure = not backend.name() == 'unitary_simulator' 
        cpmg_obj = CPMGCircuit(taus, pulse_rot, measure=measure)
        cpmg_circ = cpmg_obj.get_circuit()

        # grab parameters
        phis = cpmg_obj.phi_params

        # Make NOISE_CLASS a list if it is not already
        if not isinstance(NOISE_CLASS, list):
            NOISE_CLASS = [NOISE_CLASS]

        # Construct signal-less circuit
        if num_noise_trajs == 0:
            phi_vals = np.zeros(len(phis))
            phi_dict = {phis[i]:phi_vals[i] for i in range(len(phis))}
            bound_cpmg = cpmg_circ.bind_parameters(phi_dict)
            circuit_dict['no-sig%d' % idx] = bound_cpmg
        else:
            for traj_idx in range(num_noise_trajs):
                noise_traj_list = [NC.generate_noise(len(phis)) for NC in NOISE_CLASS]
                traj = np.sum(noise_traj_list, axis=0)
                phi_dict = {phis[i]:traj[i] for i in range(len(phis))}
                bound_cpmg = cpmg_circ.bind_parameters(phi_dict)
                circuit_dict['no-sig%d_traj%d' % (idx, traj_idx)] = bound_cpmg

        # Construct signal circuit
        for traj_idx in range(num_sig_trajs):
            sig_traj = SIGNAL_CLASS.generate_noise(len(phis))

            noise_traj_list = [NC.generate_noise(len(phis)) for NC in NOISE_CLASS]
            noise_traj = np.sum(noise_traj_list, axis=0)
            traj = sig_traj + noise_traj
            phi_dict = {phis[i]:traj[i] for i in range(len(phis))}
            bound_cpmg = cpmg_circ.bind_parameters(phi_dict)
            circuit_dict['sig%d_traj%d' % (idx, traj_idx)] = bound_cpmg

    # Run circuits
    if (backend.name() == 'qasm_simulator') or (backend.name() == 'unitary_simulator'):
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
            if backend.name() == 'unitary_simulator':
                unitary = np.asarray(results.get_unitary(circ))
                prob_0 = abs(unitary @ np.array([1, 0])[0])**2
                no_sig_prob.append(prob_0) 

            else:
                counts = results.get_counts(circ)
                zero_counts = counts.get('0',0)
                no_sig_prob.append(zero_counts/shots)
    else:
        for elem in range(len(pop)):
            traj_sum = 0
            for traj_idx in range(num_noise_trajs):
                circ = circuit_dict['no-sig%d_traj%d' % (elem, traj_idx)]
                if backend.name() == 'unitary_simulator':
                        unitary = np.asarray(results.get_unitary(circ))
                        prob_0 = abs((unitary @ np.array([1, 0]))[0])**2
                        traj_sum += prob_0
                else:                    
                    counts = results.get_counts(circ)
                    zero_counts = counts.get('0',0)
                    traj_sum += zero_counts
            # Find average cost
            if backend.name() == 'unitary_simulator':
                no_sig_prob.append(traj_sum / num_noise_trajs)
            else:
                no_sig_prob.append(traj_sum/shots/num_noise_trajs)
    no_sig_prob = np.array(no_sig_prob)

    sig_prob = []
    for elem in range(len(pop)):
        traj_sum = 0
        for traj_idx in range(num_sig_trajs):
            circ = circuit_dict['sig%d_traj%d' % (elem, traj_idx)]
            if backend.name() == 'unitary_simulator':
                unitary = np.asarray(results.get_unitary(circ))
                prob_0 = abs((unitary @ np.array([1, 0]))[0])**2
                traj_sum += prob_0
            else:
                counts = results.get_counts(circ)
                zero_counts = counts.get('0',0)
                traj_sum += zero_counts
        # Find average cost
        # sig_prob.append(traj_sum/shots/num_sig_trajs)
        if backend.name() == 'unitary_simulator':
            sig_prob.append(traj_sum / num_sig_trajs)
        else:
            sig_prob.append(traj_sum/shots/num_sig_trajs)
    sig_prob = np.array(sig_prob)

    fitness = no_sig_prob - sig_prob
    return fitness
