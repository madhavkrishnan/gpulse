# Define classes used to build the noise model and CPMG circuit

from scipy.linalg import norm
from scipy.signal import lfilter, firwin, freqz, zpk2tf
import numpy as np
from numpy.random import standard_normal
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

class Noise:
    """
    Class to handle noise and signal realisation for the CPMG circuit.

    Signal paramaters are stored in this object as well as a current realisation.
    Methods are defined to return the noise signal and build a new realisation.

    Signal is generated in two stages. A finite impulse response filter is creatd
    using scipy.signal.firwin which is applied to a signal at the central frequency
    w0.  This signal is upsampled as required by a second filter using
    scipy.signal.lfilter depending on the number of gates required for
    the CMPG pulse.

    Attributes
    ----------
    power : float
        decides the power of the signal - this is equal to the area under the
        curve of s(w).
    coeffcients : array_like
        numerator coeffcient sequence of the filter.
    length : int
        length of the initial filter
    cutoff : float
        cutoff frequency of the initial filter
    w0 : float
        central frequency of the signal
    w : array
        frequecies at which the frequency reponse of the filter is calculated
    h : array
        frequency response of the array
    upsample_length : int
        the length to which the base signal is upsampled
    noise : array
        upsampled noise signal

    Methods
    -------
    generate_noise()
        produces the upsampled noise signal.

    """

    def __init__(self, power, length, cutoff, w0, worN=512):
        """
        Parameters
        ----------
        power : float
            decides the power of the signal - this is equal to the area under the
            curve of s(w).
        length : int
            length of the initial filter
        cutoff : float
            cutoff frequency of the initial filter
        w0 : float
            central frequency of the signal
        """

        self.power = power
        self.length = length
        self.cutoff = cutoff
        self.w0 = w0
        self.worN = worN
        self.coefficients = firwin(length, cutoff) * np.cos( np.pi * w0 * np.arange(length))
        # normalises numerator so that sum(abs(h)**2)*w[1] = power
        self.coefficients = self.coefficients / norm(self.coefficients) * np.sqrt(power)
        # compute frequency response of the filter. denominator coeffcients are
        # assumed to be [1]
        self.w, self.h = freqz(self.coefficients, [1], worN=self.worN)
        self.upsample_length = None
        self.noise = None

    def generate_noise(self, upsample_length):
        """
        Generates the upsampled noise signal

        Parameters
        ----------
        upsample_length : int
            the length to which the base signal is upsampled
        """

        self.upsample_length = upsample_length

        # signal is upsampled and first 1000 transients are excluded
        self.noise =  lfilter(self.coefficients, [1], standard_normal(upsample_length + 1000))[1000:]

        # returns a copy of the signal
        return np.copy(self.noise)
    

class FAlphaNoise:

    """
    Class to generate 1/f^alpha noise
    """
    
    def __init__(self, power, alpha, gate_time=1):
        N = 2048
        wl = .001*np.pi
        wh = 0.5*np.pi
        Nf = np.ceil(2.5*(np.log10(wh)-np.log10(wl)))
        delp = (np.log10(wh)-np.log10(wl))/Nf
        logps = np.log10(wl)+.5*(1-alpha/2.)*delp + np.arange(Nf)*delp
        logzs = logps+alpha/2.*delp
        ps = 10**(logps)
        zs = 10**(logzs)
        pstx = (1-ps)/(1+ps)
        zstx = (1-zs)/(1+zs)

        b, a = zpk2tf(zstx,pstx,k=1e-4)
        w_pa,h_pa = freqz(b, a, worN=N, whole=True)
        f_nyq = 0.5*1/gate_time
        f = w_pa/np.pi*f_nyq

        pre_power = np.sum(np.abs(h_pa)**2)*f[1]
        self.coefficients = b/np.sqrt(pre_power)*np.sqrt(power)

        self.denom_coefficients = a
        self.w, self.h = freqz(self.coefficients, self.denom_coefficients, worN=N)
        self.upsample_length = None
        self.noise = None
    
    
    def generate_noise(self, upsample_length):
        """
        Generates the upsampled noise signal

        Parameters
        ----------
        upsample_length : int
            the length to which the base signal is upsampled
        """

        self.upsample_length = upsample_length

        # signal is upsampled and first 1000 transients are excluded
        self.noise =  lfilter(self.coefficients, self.denom_coefficients, standard_normal(upsample_length + 1000))[1000:]

        # returns a copy of the signal
        return np.copy(self.noise)
        

class CPMGCircuit:
    """
    Class to build and manipulate CPMG circuits.

    Attributes
    ----------
    tau : array_like
        sequence of tau values defining the interspace timing of
        the CPMG pulse.
    noise : Noise class instance
        object with signal/noise information as Attributes

    Methods
    -------
    """

    def __init__(self, taus=None, pulse_rotation=None):
        """
        Parameters
        ----------
        taus : array_like
            sequence of tau values defining the interspace timing of
            the CPMG pulse.
        pulse_rotation : array_like or None, optional
            CPMG pulse roation angles for each cycle. Default value is pi

        """
        if taus == None:
            self.taus = [1] * 5
        else:
            self.taus = taus
        # computes the number of signal gates needed for the CMPG sequence
        # the number of rz rotations in the circuit
        rz_count = 4 * sum(self.taus)
        # parameter vector (rz rotation angles)
        self.phi_params = ParameterVector("phi", rz_count)
        # self.noise = noise
        # # generates the upsampled noise signal
        # self.noise.generate_noise(self.rz_count)

        # defines the rotation angle for each pulse cycle
        if pulse_rotation is None:
            self.pulse_rotation = [np.pi]*len(self.taus)
        else:
            self.pulse_rotation = [2*np.pi/theta for theta in pulse_rotation]
#         # initialise circuit library
#         # self.circuit_lib = {}
        
#         # Define signal/noise
#         if signal == None:
#             self.signal = np.zero(len(self.taus))
#         else:
#             self.signal = signal
#         if noise == None:
#             self.noise = np.zero(len(self.taus))
#         else:
#             self.noise = noise
#         self.dephasing_rot = self.signal + self.noise

        # builds the parameterised CMPG circuit
        self.circuit = self.build_circuit(self.taus, self.pulse_rotation)

        # binds the signal instance to the circuit
        # self.circuit = self.bind_circuit(self.unbound_circuit, self.noise)

    def get_circuit(self):
        return self.circuit.copy()

    def build_circuit(self, taus, pulse_rotation):
        """
        Function to build the CMPG circuit.

        Parameters
        ----------
        taus : array_like
            sequence of tau values defining the interspace timing of
            the CPMG pulse.

        Returns
        -------
        circuit : Qiskit.QuantumCircuit
            Parameterisd circuit with signal and CPMG pulses
        """

        # build unbound circuit key
        # circ_key = [str(tau) for tau in taus]
        # circ_key = "tau_" + "_".join(circ_key)
        #
        # # check if circuit already exists in the library
        # if circ_key in self.circuit_lib:
        #     self.circuit = self.circuit_lib[circ_key].copy()
        #     self.taus = taus
        #     return self.circuit.copy()
        #
        # # Regenerates signal if taus has changed
        # if not taus == self.taus:
        #     self.taus = taus
        #     self.rz_count = 4 * sum(self.taus)
        #     self.noise.generate_noise(self.rz_count)
        #     self.phi_params = ParameterVector("phi", self.rz_count)

        self.taus = taus

        # TODO: Add pulse rotation specification to build_circuit.
        #self.pulse_rotation = [np.pi] * len(self.taus)
        self.pulse_rotation = pulse_rotation

        rz_count = 4 * sum(self.taus)
        self.phi_params = ParameterVector("phi", rz_count)
        # builds circuit
        circuit = QuantumCircuit(1,1)

        # Encoding in y basis
        circuit.ry(np.pi / 2, 0)

        # initialise indices to track gate paramaters
        phi_counter = 0
        control_counter = 0
        # Building unbound parametrised circuit
        for tau in self.taus:
            # gates before first control pulse
            for idx in range(tau):
                circuit.i(0)
                circuit.rz(self.phi_params[phi_counter], 0)
                phi_counter += 1

            # first control pulse
            circuit.rx(self.pulse_rotation[control_counter], 0)

            # gates after first pulse
            for idx in range(2*tau):
                circuit.i(0)
                circuit.rz(self.phi_params[phi_counter], 0)
                phi_counter += 1

            # second control pulse
            circuit.rx(self.pulse_rotation[control_counter], 0)

            # gates after second pulse
            for idx in range(tau):
                circuit.i(0)
                circuit.rz(self.phi_params[phi_counter], 0)
                phi_counter += 1
            # update pulse counter
            control_counter += 1

        # Decoding in y basis
        circuit.ry(-np.pi / 2, 0)
        circuit.measure(0, 0)

        # self.circuit_lib[circ_key] = circuit.copy()

        self.circuit = circuit.copy()
        return circuit
    
    
    
class RxRyCircuit(CPMGCircuit):
    
    
    def build_circuit(self, taus, pulse_rotation):
        """
        Function to build RxRy circuit.

        Parameters
        ----------
        taus : array_like
            sequence of tau values defining the interpulse timing 
            
        pulse_rotations : array_like
            sequence of angle values defining the rotation angle for
            the control pulse

        Returns
        -------
        circuit : Qiskit.QuantumCircuit
            Parameterisd circuit with signal and RxRy pulses
        """

        self.taus = taus

        self.pulse_rotation = pulse_rotation

        rz_count = 4 * sum(self.taus)
        self.phi_params = ParameterVector("phi", rz_count)
        # builds circuit
        circuit = QuantumCircuit(1,1)

        # Encoding in y basis
        circuit.ry(np.pi / 2, 0)

        # initialise indices to track gate paramaters
        phi_counter = 0
        control_counter = 0
        # Building unbound parametrised circuit
        for tau in self.taus:
            # gates before first control pulse
            for idx in range(tau):
                circuit.i(0)
                circuit.rz(self.phi_params[phi_counter], 0)
                phi_counter += 1

            # first control pulse
            circuit.rx(self.pulse_rotation[control_counter], 0)

            # gates after first pulse
            for idx in range(2*tau):
                circuit.i(0)
                circuit.rz(self.phi_params[phi_counter], 0)
                phi_counter += 1

            # second control pulse
            circuit.rx(self.pulse_rotation[control_counter], 0)

            # gates after second pulse
            for idx in range(tau):
                circuit.i(0)
                circuit.rz(self.phi_params[phi_counter], 0)
                phi_counter += 1
            # update pulse counter
            control_counter += 1

        # Decoding in y basis
        circuit.ry(-np.pi / 2, 0)
        circuit.measure(0, 0)

        # self.circuit_lib[circ_key] = circuit.copy()

        self.circuit = circuit.copy()
        return circuit