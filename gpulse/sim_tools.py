# single qubit simulator
from numpy import array, pi, sin, cos , exp, random
from gpulse.util import unpack


theta = pi
# define gates
def rx(theta):
    return array( [[cos(theta / 2), -1j * sin(theta / 2) ], [-1j * sin(theta / 2), cos(theta / 2)]] )

def ry(theta):
    return array( [[cos(theta / 2), -sin(theta / 2) ], [sin(theta / 2), cos(theta / 2)]] )

def rz(theta):
    return array( [[exp(-1j * theta / 2), 0 ], [0, exp(1j * theta / 2)]] )

def qubit_sim(z_rot, pulse, return_prob = False, ansatz="CPMG"):
    """Qubit simulator for a pulse ansatz"""
    
    if ansatz != "CPMG":
        raise("Only CPMG ansatz supported")

    # Initialise state
    state = ry(pi/2) @ array([[1], [0]])

    # Apply Rz and Rx gates.
    z_idx = 0
    for tau, xrot in pulse:

        # Compute z_rotation angles
        z_theta1 = sum(z_rot[z_idx: z_idx+tau])
        z_theta2 = sum(z_rot[z_idx+tau: z_idx+3*tau])
        z_theta3 = sum(z_rot[z_idx+3*tau: z_idx+4*tau])
        
        #update z rotation index
        z_idx += 4*tau

        #update state
        state = rz(z_theta1) @ state
        state = rx(2 * pi / xrot) @ state
        state = rz(z_theta2) @ state
        state = rx(2 * pi / xrot) @ state
        state = rz(z_theta3) @ state
    
    # Decode
    state = ry(-pi/2) @ state

    if return_prob:
        return abs(state[0][0])**2

    return state



# Ideal circuit based on binomial distribution. 
def circuit_sim(z_rot, pulse, shots, ansatz="CPMG"):

    state_out = qubit_sim(z_rot, pulse, ansatz="CPMG")
    p0 = abs(state_out[0])**2

    return random.binomial(shots, p0)
