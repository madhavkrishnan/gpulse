# Utility functions
from gpulse import Noise

def unpack(pulse_sequence):
    """ Helper function to unpack and individual [(tau1, rot1), (tau2, rot2),...] 
    -> [tau1, tau2,...] , [rot1, rot2,...]"""
    unzipped_object = zip(*pulse_sequence)
    return list(unzipped_object)


def get_sgen(signal_power, target_tau, length = 256, cutoff=0.01):
    """
    Helper function to generate a signal with central frequence 1 / (2 * traget_tau)
    """
  
    # Target frequency
    w0 =  1 / (2 * target_tau)

    return Noise(signal_power, length, cutoff, w0)

def get_rzcount(pulse_sequence):
    taus, rots = unpack(pulse_sequence)
    
    return(4 * sum(taus))
