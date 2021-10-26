# This file contains different noise models
from numpy import pi, trapz

def lorenzian(w, w0, gamma, power):
    """
    Generate a lorentian frequncy spectrum

    w : array
        frequency values to calculate the distribution for

    w0 : float
        center of the distirbution

    gamma : float
        full width at half maximum

    power: float
        area under the curive * 2pi

    """
    gamma *= 0.5

    l = (gamma / pi) / ((w-w0)**2 + gamma**2)



    norm = trapz(l, w)

    l *=   power *   pi / norm

    return l
