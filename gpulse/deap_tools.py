# File contains tools for deap for individuals with variable pulse spacing
# and rotation angles.

from deap import tools, base
import random
from gpulse.util import unpack


def snip(ind):
    """
    Random deletion of last r cycles of a
    """
    cut = random.randint(1,len(ind))
    del ind[cut:]
    return ind

def init_ind(*args):
    """
    Helper function to initialise a population.

    The function is set up  

    Parameters
    ----------
    args :

    """
    tau_lims, rot_lims = args
    tau_min, tau_max = tau_lims
    rot_min, rot_max = rot_lims

    tau = random.randint(tau_min, tau_max)
    rot = random.randint(rot_min, rot_max)

    return [tau, rot]

def mutate(ind, 
           tau_lims=[1,10], 
           rot_lims=[1,10], 
           indpb=0.5, 
           vary_length=False, 
           GPB=0.5,
           max_time = None):


    # unzipped_object = zip(*ind)
    # unzipped_list = [list(i) for i in unzipped_object]

    # taus, rots  = unzipped_list
    taus, rots = unpack(ind)

    if len(taus) != len(rots):
        raise ValueError("Length of taus must equal length of rots")

    # mutate
    tools.mutUniformInt(taus, low=tau_lims[0], up = tau_lims[1], indpb=indpb)
    tools.mutUniformInt(rots, low=rot_lims[0], up = rot_lims[1], indpb=indpb)

    new_pulse = [list(i) for i in zip(taus, rots)]

    # for i in range(len(ind)):
    #     ind[i] = new_pulse[i]

    if vary_length:
        r = random.random()
        # mutate length only half the time
        if (r < GPB) and (r > GPB / 2):
            if len(new_pulse) > 1:
                new_pulse.pop(-1)
        elif r < GPB:
            new_ele = [random.randint(*tau_lims), random.randint(*rot_lims)]
            new_pulse.append(new_ele)
    
    # Check time constraint, return unchanged if exceeding total time. 
    if max_time is not None:
        tnew, rnew = unpack(new_pulse)
        if sum(tnew) > max_time:
            return ind, 
    ind[:] = new_pulse[:]

    return ind,

def cx(ind_1, ind_2, max_time=None):

    if (len(ind_1) == 1) or (len(ind_2) == 1):
        return ind_1, ind_2

    # unzipped_object_1 = zip(*ind_1)
    # unzipped_object_2 = zip(*ind_2)

    # unzipped_list_1 = [list(i) for i in unzipped_object_1]
    # unzipped_list_2 = [list(i) for i in unzipped_object_2]

    # taus_1, rots_1  = unzipped_list_1
    # taus_2, rots_2  = unzipped_list_2
    
    # taus_1, rots_1 = unpack(ind_1)
    # taus_2, rots_2 = unpack(ind_2)

    # crossover
    pulse_1 = ind_1[:]
    pulse_2 = ind_2[:]
    tools.cxTwoPoint(pulse_1, pulse_2)

    # Check time constraint, return unchanged if exceeding max time
    if max_time is not None:
        t1, r1 = unpack(pulse_1)
        t2, r2 = unpack(pulse_2)
        if (sum(t1) > max_time) or (sum(t2) > max_time):
            return ind_1, ind_2

    # tools.cxTwoPoint(taus_1, taus_2)
    # tools.cxTwoPoint(rots_1, rots_2)
    # new_pulse_1 = [list(i) for i in zip(taus_1, rots_1)]
    # new_pulse_2 = [list(i) for i in zip(taus_2, rots_2)]

    ind_1[:] = pulse_1[:]
    ind_2[:] = pulse_2[:]

    # for i in range(len(ind_1)):
    #     ind_1[i] = new_pulse_1[i]

    # for i in range(len(ind_2)):
    #     ind_2[i] = new_pulse_2[i]

    return ind_1, ind_2

# print( mutate(list(zip([5]*10, [2]*10)), tau_lims=[1,10] , rot_lims= [1,10], indpb=0.5))
