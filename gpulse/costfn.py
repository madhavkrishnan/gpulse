# This file contains different cost functions for the GA optimiser
from gpulse.ffn import decay_probability, chi
import numpy as np
from numpy.random import binomial
from gpulse.CPMG import CPMGCircuit
import qiskit as qk

# This no longer works with the latest version of qiskit  
# from qiskit.providers.ibmq.managed import IBMQJobManager

from gpulse.util import unpack, get_rzcount
from gpulse.sim_tools import circuit_sim, qubit_sim


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




def noise_average(pulse_list, noise_gen, signal_gen = None, trajs=[1]):
    """
    A list of pulses and noise & signal generators are taken as input. p0 is 
    computed for all pulses in the list and over number of trajectories 
    in trajs. Computes both noise and noise + signal cases if signal gen
    is provided.
    """

    p0_noise_list = []
    p0_signoise_list = []

    for t in trajs:
        rz_counts = [get_rzcount(p) for p in pulse_list]

        # Generate t number of noise realisations for each pulse seq
        # ordered as [[ p1_traj1, p1_traj2,..], [[ p2_traj1, p2_traj2,..], ...]
        noise  =[[noise_gen.generate_noise(r) for i in range(t)] for r in rz_counts] 

        p0_noise = [[qubit_sim(nr, p, return_prob=True) for nr in n] for n,p in zip(noise, pulse_list)]

        p0_noise = [np.mean(p) for p in p0_noise]
               
        p0_noise_list.append(p0_noise)

        if signal_gen is None:
            continue
        
        signal = [[signal_gen.generate_noise(r) for i in range(t)] for r in rz_counts]

        signoise = [[sr + nr for sr, nr in zip(s, n) ] for s,n in zip(signal, noise) ]
        
        p0_signoise =  [[qubit_sim(signr, p, return_prob=True) for signr in n] for n,p in zip(signoise, pulse_list)]
        
        p0_signoise = [np.mean(p) for p in p0_signoise]
          
        p0_signoise_list.append(p0_signoise)
    
    if len(p0_noise_list) == 1:
        p0_noise_list = p0_noise_list[0]
        if signal_gen is not None:
            p0_signoise_list = p0_signoise_list[0]

    if signal_gen is None:
        return p0_noise_list
    return p0_noise_list, p0_signoise_list    


def cost_sig_noise_qsim(noise_gen, signal_gen, pop, shots=False, trajs=1):


    p0_n, p0_nsig = noise_average(pop, noise_gen, signal_gen=signal_gen, trajs=[trajs]) 

    fit = [n - ns for n, ns in zip(p0_n, p0_nsig)]

    return fit


    # # loop over population
    # fit = []
    # for pulse_seq in pop:
        
    #     rz_count = get_rzcount(pulse_seq)
        
    #     # Generate noise and signal rotations
    #     noise = [NOISE_CLASS.generate_noise(rz_count) for i in range(trajs)]
    #     signal = [SIGNAL_CLASS.generate_noise(rz_count) for i in range(trajs)]

    #     # no-signal final state
    #     no_sig_prob = sum([ abs(qubit_sim(n, pulse_seq)[0])**2 for n in noise]) / trajs 

    #     # signal final state
    #     sig_prob = sum([ abs(qubit_sim(np.array(n) + np.array(s), pulse_seq)[0])**2 for n,s in zip(noise, signal)]) / trajs

    #     fit.append( (no_sig_prob - sig_prob)[0])
    # return fit





        

